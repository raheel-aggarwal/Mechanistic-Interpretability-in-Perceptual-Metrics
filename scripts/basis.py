"""
basis.py
--------
1-D and 2-D orthonormal basis construction: DCT-II and generalised Haar.

Public API
----------
dct_basis(N)          -> np.ndarray (N, N)
haar_basis(N)         -> np.ndarray (N, N)
get_1d_basis(N, type) -> np.ndarray (N, N)
build_2d_basis(N, type) -> np.ndarray (N, N, N, N)
"""

import math
from typing import List

import numpy as np


# ── 1-D DCT-II ────────────────────────────────────────────────────────────────

def dct_basis(N: int) -> np.ndarray:
    """
    Orthonormal DCT-II basis for R^N.

    U[k, n] = alpha_k * cos(pi*(2n+1)*k / (2N))
    alpha_0 = 1/sqrt(N),  alpha_k = sqrt(2/N) for k > 0

    Returns
    -------
    U : ndarray (N, N)  — U[k] is the k-th basis vector (row).
    Satisfies  U @ U.T == I_N  (orthonormal).
    """
    ns = np.arange(N)
    U  = np.zeros((N, N))
    for k in range(N):
        alpha = math.sqrt(1.0 / N) if k == 0 else math.sqrt(2.0 / N)
        U[k]  = alpha * np.cos(math.pi * (2 * ns + 1) * k / (2 * N))
    return U


# ── 1-D generalised Haar ──────────────────────────────────────────────────────

def haar_basis(N: int) -> np.ndarray:
    """
    Generalised Haar basis for R^N using recursive binary splitting.

    Ordering: scaling vector u0 first, then depth-first parent-before-children
    traversal (parent wavelet → left subtree → right subtree).

    Returns
    -------
    U : ndarray (N, N)  — U[k] is the k-th basis vector (row).
    Satisfies  U @ U.T == I_N  (orthonormal).
    """
    vectors: List[np.ndarray] = []

    # scaling vector (DC component)
    vectors.append(np.ones(N) / math.sqrt(N))

    def _recurse(lo: int, hi: int) -> None:
        length = hi - lo + 1
        if length < 2:
            return
        left_size  = length // 2
        right_size = length - left_size

        v = np.zeros(N)
        v[lo : lo + left_size]   =  1.0
        v[lo + left_size : hi+1] = -left_size / right_size
        vectors.append(v / np.linalg.norm(v))

        _recurse(lo, lo + left_size - 1)
        _recurse(lo + left_size, hi)

    _recurse(0, N - 1)
    return np.array(vectors)   # (N, N)


# ── Convenience wrappers ──────────────────────────────────────────────────────

def get_1d_basis(N: int, basis_type: str = "haar") -> np.ndarray:
    """
    Return the 1-D orthonormal basis matrix of shape (N, N).

    Parameters
    ----------
    basis_type : "haar" | "dct"
    """
    if basis_type == "haar":
        return haar_basis(N)
    if basis_type == "dct":
        return dct_basis(N)
    raise ValueError(f"Unknown basis_type {basis_type!r}. Use 'haar' or 'dct'.")


def build_2d_basis(N: int, basis_type: str = "haar") -> np.ndarray:
    """
    Build the 2-D orthonormal basis over R^{N×N} by outer product.

    Psi[p, q, x, y] = U[p, x] * U[q, y]

    Returns
    -------
    Psi : ndarray (N, N, N, N)
    """
    U = get_1d_basis(N, basis_type)
    return np.einsum("pi,qj->pqij", U, U)   # (N, N, N, N)


# ── Self-test (run as script) ─────────────────────────────────────────────────

def _run_tests() -> None:
    for btype, fn in [("DCT", dct_basis), ("Haar", haar_basis)]:
        for N in (3, 5, 11):
            U   = fn(N)
            err = np.max(np.abs(U @ U.T - np.eye(N)))
            status = "✓" if err < 1e-12 else "✗"
            print(f"{btype} N={N:2d}  ||U U^T - I||_inf = {err:.2e}  {status}")

    for N in (3, 5):
        Psi = build_2d_basis(N, "haar")
        M   = Psi.reshape(N**2, N**2)
        err = np.max(np.abs(M @ M.T - np.eye(N**2)))
        status = "✓" if err < 1e-12 else "✗"
        print(f"2D Haar N={N}  ||Ψ Ψ^T - I||_inf = {err:.2e}  {status}")


if __name__ == "__main__":
    _run_tests()
