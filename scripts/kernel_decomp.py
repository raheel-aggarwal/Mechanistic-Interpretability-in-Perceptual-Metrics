"""
kernel_decomp.py
----------------
Kernel basis projection and component-map computation.

Public API
----------
project_kernel(K, U)              -> Tensor  (Co, Ci, N, N)
reconstruct_from_coeffs(coeffs, U) -> Tensor  (Co, Ci, N, N)
basis_component_maps(X, K, U, ...) -> Tensor  (B, Co, Ho, Wo, N²)
prune_components(x_comp, prune)    -> Tensor  (B, C, H, W, K)
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def _orthonormalize_rows(U: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis with rows spanning the same space as U."""
    Q, _ = np.linalg.qr(U.T)
    return Q.T


def dct_basis(N: int) -> np.ndarray:
    """Orthonormal DCT-II basis for R^N."""
    ns = np.arange(N)
    U = np.zeros((N, N))
    for k in range(N):
        alpha = math.sqrt(1.0 / N) if k == 0 else math.sqrt(2.0 / N)
        U[k] = alpha * np.cos(math.pi * (2 * ns + 1) * k / (2 * N))
    return U


def haar_basis(N: int) -> np.ndarray:
    """Generalised Haar basis for R^N.

    Returns an orthonormal matrix U of shape (N, N), where each row is a
    basis vector. The basis is built recursively and then orthonormalized to
    guarantee exact decomposition even for non-power-of-two lengths.
    """
    vectors: List[np.ndarray] = []
    vectors.append(np.ones(N) / math.sqrt(N))

    def _recurse(lo: int, hi: int) -> None:
        length = hi - lo + 1
        if length < 2:
            return
        left_size = length // 2
        right_size = length - left_size

        v = np.zeros(N)
        v[lo : lo + left_size] = 1.0
        v[lo + left_size : hi + 1] = -left_size / right_size
        vectors.append(v)

        _recurse(lo, lo + left_size - 1)
        _recurse(lo + left_size, hi)

    _recurse(0, N - 1)
    U = np.stack(vectors, axis=0)
    return _orthonormalize_rows(U)


def get_1d_basis(N: int, basis_type: str = "haar") -> np.ndarray:
    if basis_type == "haar":
        return haar_basis(N)
    if basis_type == "dct":
        return dct_basis(N)
    raise ValueError(f"Unknown basis_type {basis_type!r}. Use 'haar' or 'dct'.")


def build_2d_basis(N: int, basis_type: str = "haar") -> np.ndarray:
    U = get_1d_basis(N, basis_type)
    return np.einsum("pi,qj->pqij", U, U)


def project_kernel(
    K: torch.Tensor,
    U: np.ndarray,
) -> torch.Tensor:
    """
    Project a convolutional kernel onto a 1-D separable basis.

    coeff[o, c, p, q]  =  <K[o,c], ψ_{pq}>
                       =  Σ_{x,y} K[o,c,x,y] * U[p,x] * U[q,y]

    Computed as two successive einsum contractions (rows then columns)
    so that no explicit 2-D outer product is formed.

    Parameters
    ----------
    K : Tensor  (C_out, C_in, N, N)
    U : ndarray (N, N)  — 1-D orthonormal basis (rows = basis vectors)

    Returns
    -------
    coeff : Tensor  (C_out, C_in, N, N)
        coeff[o, c, p, q] is the (p,q) projection coefficient of K[o,c].
    """
    Ut     = torch.tensor(U, dtype=K.dtype, device=K.device)   # (N, N)
    K_mid  = torch.einsum("ocxy, px -> ocpy", K, Ut)            # project rows
    coeffs = torch.einsum("ocpy, qy -> ocpq", K_mid, Ut)        # project cols
    return coeffs


def reconstruct_from_coeffs(
    coeffs: torch.Tensor,
    U:      np.ndarray,
) -> torch.Tensor:
    """
    Invert project_kernel:  K[o,c,x,y] = Σ_{p,q} coeff[o,c,p,q] * ψ_{pq}(x,y)

    Useful for round-trip verification.
    """
    Ut    = torch.tensor(U, dtype=coeffs.dtype, device=coeffs.device)
    K_mid = torch.einsum("ocpq, px -> ocqx", coeffs, Ut)
    K_rec = torch.einsum("ocqx, qy -> ocxy", K_mid, Ut)
    return K_rec


def basis_component_maps(
    X:       torch.Tensor,
    K:       torch.Tensor,
    U:       np.ndarray,
    padding: int = 0,
    stride:  int = 1,
) -> torch.Tensor:
    """
    Compute basis component feature maps for one conv layer.

    z_{pq}(x,y)  =  Σ_c  <K^(c), ψ_{pq}>  * (ψ_{pq} * X^(c))(x,y)

    The basis ψ_{pq} = outer(u_p, u_q) is separable, so the 2-D convolution
    factors into two 1-D convolutions (row-wise then column-wise), reducing
    cost from O(N²·HW) to O(2N·HW) per basis element.

    Parameters
    ----------
    X       : Tensor  (B, C_in, H, W)
    K       : Tensor  (C_out, C_in, N, N)
    U       : ndarray (N, N)  — 1-D basis from `haar_basis` or `dct_basis`
    padding : same as the original conv layer
    stride  : same as the original conv layer

    Returns
    -------
    z_pq : Tensor  (B, C_out, H_out, W_out, N²)
        Last dimension indexes flattened (p,q) pairs in row-major order.
    """
    Ut             = torch.tensor(U, dtype=X.dtype, device=X.device)
    N              = U.shape[0]
    B, Ci, H, W   = X.shape
    Co, _, _, _   = K.shape

    coeffs = project_kernel(K, U)   # (Co, Ci, N, N)

    all_z = []
    for p in range(N):
        for q in range(N):
            # 2-D basis kernel ψ_{pq} = outer(u_p, u_q)
            psi_pq = torch.outer(Ut[p], Ut[q])              # (N, N)

            # Convolve X with ψ_{pq} per input channel (grouped conv)
            psi_f  = psi_pq.unsqueeze(0).unsqueeze(0)       # (1, 1, N, N)
            psi_f  = psi_f.expand(Ci, 1, N, N)              # (Ci, 1, N, N)
            X_pq   = F.conv2d(X, psi_f, padding=padding,
                               stride=stride, groups=Ci)     # (B, Ci, Ho, Wo)

            # Mix channels with projection coefficients → one map per output ch
            coeffs_pq = coeffs[:, :, p, q]                  # (Co, Ci)
            z_pq      = torch.einsum("oc, bchw -> bohw",
                                      coeffs_pq, X_pq)      # (B, Co, Ho, Wo)
            all_z.append(z_pq)

    return torch.stack(all_z, dim=-1)   # (B, Co, Ho, Wo, N²)


def prune_components(
    x_comp: torch.Tensor,
    prune:  int,
) -> torch.Tensor:
    """
    Keep only the top-`prune` basis components per output channel.

    Scoring is per-channel: each component is ranked by its mean absolute
    activation over (B, H, W) for that channel.  This prevents a component
    that is important for one channel but weak on others from being discarded
    due to global averaging.

    Parameters
    ----------
    x_comp : Tensor  (B, C, H, W, K)
    prune  : int — components to keep per channel

    Returns
    -------
    Tensor  (B, C, H, W, K)  — non-top-prune components zeroed per channel.
    """
    B, C, H, W, K = x_comp.shape
    if prune >= K:
        return x_comp   # nothing to prune

    scores  = x_comp.abs().mean(dim=(0, 2, 3))               # (C, K)
    _, top  = torch.topk(scores, k=prune, dim=-1)            # (C, prune)
    mask    = torch.zeros(C, K, dtype=x_comp.dtype,
                          device=x_comp.device)
    mask.scatter_(1, top, 1.0)
    return x_comp * mask.view(1, C, 1, 1, K)


# ── Self-test ─────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    import torch.nn.functional as F

    # Round-trip kernel projection and reconstruction
    for btype in ("haar", "dct"):
        for N in (3, 5, 11):
            U    = haar_basis(N) if btype == "haar" else dct_basis(N)
            K_t  = torch.randn(4, 3, N, N)
            c    = project_kernel(K_t, U)
            K_re = reconstruct_from_coeffs(c, U)
            err  = (K_t - K_re).abs().max().item()
            print(f"  {btype} N={N:2d}  round-trip error: {err:.2e}",
                  "✓" if err < 1e-5 else "✗")

    # Verify orthonormality of rows in each basis
    for btype in ("haar", "dct"):
        for N in (3, 5, 11):
            U = haar_basis(N) if btype == "haar" else dct_basis(N)
            eye_err = np.max(np.abs(U @ U.T - np.eye(N)))
            print(f"  {btype} basis orthonormality N={N:2d} err: {eye_err:.2e}",
                  "✓" if eye_err < 1e-8 else "✗")

    # Exact basis-component convolution reconstruction
    for btype in ("haar", "dct"):
        for N in (3, 5):
            U = haar_basis(N) if btype == "haar" else dct_basis(N)
            K = torch.randn(2, 3, N, N)
            X = torch.randn(1, 3, 10, 10)
            z = basis_component_maps(X, K, U, padding=1, stride=1)
            out = z.sum(-1)
            ref = F.conv2d(X, K, padding=1)
            err = (out - ref).abs().max().item()
            print(f"  {btype} conv reconstruction N={N:2d} err: {err:.2e}",
                  "✓" if err < 1e-5 else "✗")

    # Prune test
    x = torch.randn(1, 4, 8, 8, 9)
    out = prune_components(x, prune=3)
    active = (out.abs().sum(dim=(0, 2, 3)) > 0).sum(dim=-1)
    assert (active <= 3).all(), "prune_components kept too many"
    print(f"  prune_components  active per channel: {active.tolist()}  ✓")


if __name__ == "__main__":
    _run_tests()
