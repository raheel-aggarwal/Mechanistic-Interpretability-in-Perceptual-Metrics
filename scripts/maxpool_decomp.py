"""
maxpool_decomp.py
-----------------
MaxPool decomposition with two modes:
- 'softmax': softmax-approximated additive decomposition with adaptive α (default)
- 'max_pixel': direct pass-through from the max pixel in each pooling window

Public API
----------
estimate_alpha(p_original, kernel_size, stride, ...)          -> float
maxpool_decompose(p_components, p_original, kernel_size, ...) -> Tensor (softmax mode)
maxpool_decompose_max_pixel(p_components, p_original, ...)    -> Tensor (max pixel mode)
"""

import numpy as np
import torch
import torch.nn.functional as F


def estimate_alpha(
    p_original:         torch.Tensor,
    kernel_size:        int,
    stride:             int,
    target_gap_product: float = 30.0,
    alpha_max:          float = 200.0,
    alpha_fallback:     float = 50.0,
    overflow_threshold: float = 600.0,   # < 700 for float32 safety
) -> float:
    """
    Estimate a numerically safe α for the softmax MaxPool approximation.

    Strategy
    --------
    We want  α * gap >> 1  (approximation accuracy) while
             α * ||p||_inf < overflow_threshold  (no float32 overflow).

    1. Estimate median gap = p_max - second_max over all pooling windows.
    2. Set α = target_gap_product / median_gap.
    3. Clamp so α * max_activation < overflow_threshold.
    4. Clamp to [1, alpha_max].

    Parameters
    ----------
    p_original          : Tensor (B, C, H, W)
    target_gap_product  : desired α * gap  (default 30 → error < 1e-13)
    alpha_max           : hard ceiling
    alpha_fallback      : used when gap cannot be estimated (near ties)
    overflow_threshold  : α * ||p||_inf must stay below this

    Returns
    -------
    float — recommended α for this pooling layer.
    """
    with torch.no_grad():
        ks = kernel_size
        windows = p_original.unfold(-2, ks, stride).unfold(-1, ks, stride)
        windows = windows.reshape(-1, ks * ks)   # (n_windows, k²)

        if windows.numel() == 0:
            return alpha_fallback

        top2, _ = torch.topk(windows, k=min(2, windows.shape[-1]), dim=-1)
        if top2.shape[-1] < 2:
            return alpha_fallback

        gaps       = (top2[:, 0] - top2[:, 1]).abs()
        median_gap = float(gaps.median())

        if median_gap < 1e-8:
            alpha = alpha_fallback
        else:
            alpha = target_gap_product / median_gap

        p_inf = float(p_original.abs().max())
        if p_inf > 1e-8:
            alpha = min(alpha, overflow_threshold / p_inf)

        alpha = float(np.clip(alpha, 1.0, alpha_max))

    return alpha


def maxpool_decompose(
    p_components: torch.Tensor,
    p_original:   torch.Tensor,
    kernel_size:  int,
    stride:       int,
    padding:      int   = 0,
    alpha:        float = 100.0,
) -> torch.Tensor:
    """
    Additively decompose MaxPool via the softmax approximation.

    Scalar form:
        MaxPool(p_1, ..., p_k)  ≈  Σ_i  p_i * softmax(α p)_i

    With basis decomposition p_i = Σ_j p_{ij}:
        MaxPool  ≈  Σ_j  Σ_i  p_{ij} * softmax(α p)_i

    Numerically stable: softmax computed after subtracting max(p) (log-sum-exp).

    Parameters
    ----------
    p_components : Tensor  (B, C, H, W, K)  — K basis components pre-pooling
    p_original   : Tensor  (B, C, H, W)     — grounding activations pre-pooling
    kernel_size, stride, padding : pooling geometry
    alpha        : softmax sharpness (use estimate_alpha for auto-selection)

    Returns
    -------
    Tensor  (B, C, H_out, W_out, K)
    """
    B, C, H, W, K = p_components.shape
    ks = kernel_size

    if padding > 0:
        p_original   = F.pad(p_original, [padding] * 4)
        pc_bckw      = p_components.permute(0, 1, 4, 2, 3).reshape(B, C * K, H, W)
        pc_bckw      = F.pad(pc_bckw, [padding] * 4)
        H_pad, W_pad = pc_bckw.shape[-2], pc_bckw.shape[-1]
        p_components = pc_bckw.view(B, C, K, H_pad, W_pad).permute(0, 1, 3, 4, 2)

    H_in, W_in = p_original.shape[-2], p_original.shape[-1]
    H_out = (H_in - ks) // stride + 1
    W_out = (W_in - ks) // stride + 1
    L     = H_out * W_out

    # Unfold original activations → (B, C, L, ks²)
    p_unf = F.unfold(p_original, kernel_size=ks, stride=stride)    # (B, C*ks², L)
    p_win = p_unf.view(B, C, ks * ks, L).permute(0, 1, 3, 2)       # (B, C, L, ks²)

    # Numerically stable softmax weights
    shift   = p_win.max(dim=-1, keepdim=True).values
    weights = F.softmax(alpha * (p_win - shift), dim=-1)             # (B, C, L, ks²)

    # Unfold components: merge (C, K) → single channel dim → unfold → restore
    pc_bck = p_components.permute(0, 1, 4, 2, 3).reshape(B, C * K, H_in, W_in)
    pc_unf = F.unfold(pc_bck, kernel_size=ks, stride=stride)         # (B, C*K*ks², L)
    pc_win = pc_unf.view(B, C, K, ks * ks, L).permute(0, 1, 4, 3, 2)# (B, C, L, ks², K)

    # Contract weights with components
    z_out = (pc_win * weights.unsqueeze(-1)).sum(dim=-2)              # (B, C, L, K)
    return z_out.view(B, C, H_out, W_out, K)


def maxpool_decompose_max_pixel(
    p_components: torch.Tensor,
    p_original:   torch.Tensor,
    kernel_size:  int,
    stride:       int,
    padding:      int = 0,
) -> torch.Tensor:
    """
    Pass through decomposition from the actual max pixel in each pooling window.

    For each window, identify the pixel with maximum activation value in p_original,
    and pass through only its decomposed components without any weighting. All other
    pixels in the window are ignored.

    Parameters
    ----------
    p_components : Tensor  (B, C, H, W, K)  — K basis components pre-pooling
    p_original   : Tensor  (B, C, H, W)     — grounding activations pre-pooling
    kernel_size, stride, padding : pooling geometry

    Returns
    -------
    Tensor  (B, C, H_out, W_out, K)
    """
    B, C, H, W, K = p_components.shape
    ks = kernel_size

    if padding > 0:
        p_original   = F.pad(p_original, [padding] * 4)
        pc_bckw      = p_components.permute(0, 1, 4, 2, 3).reshape(B, C * K, H, W)
        pc_bckw      = F.pad(pc_bckw, [padding] * 4)
        H_pad, W_pad = pc_bckw.shape[-2], pc_bckw.shape[-1]
        p_components = pc_bckw.view(B, C, K, H_pad, W_pad).permute(0, 1, 3, 4, 2)

    H_in, W_in = p_original.shape[-2], p_original.shape[-1]
    H_out = (H_in - ks) // stride + 1
    W_out = (W_in - ks) // stride + 1
    L     = H_out * W_out

    # Unfold original activations → (B, C, L, ks²)
    p_unf = F.unfold(p_original, kernel_size=ks, stride=stride)    # (B, C*ks², L)
    p_win = p_unf.view(B, C, ks * ks, L).permute(0, 1, 3, 2)       # (B, C, L, ks²)

    # Find argmax per window: (B, C, L, 1)
    argmax_idx = torch.argmax(p_win, dim=-1, keepdim=True)          # (B, C, L, 1)

    # Unfold components and extract max pixel components
    pc_bck = p_components.permute(0, 1, 4, 2, 3).reshape(B, C * K, H_in, W_in)
    pc_unf = F.unfold(pc_bck, kernel_size=ks, stride=stride)         # (B, C*K*ks², L)
    pc_win = pc_unf.view(B, C, K, ks * ks, L).permute(0, 1, 4, 3, 2)# (B, C, L, ks², K)

    # Gather max pixel components: expand argmax to K dim, gather
    argmax_expanded = (argmax_idx.unsqueeze(-1).expand(-1,-1,-1,1,K))    # (B, C, L, K, 1)
    z_out = torch.gather(pc_win, dim=-2, index=argmax_expanded).squeeze(-2)  # (B, C, L, K)
    return z_out.view(B, C, H_out, W_out, K)


# ── Self-test ─────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    B, C, H, W, K = 1, 2, 6, 6, 9
    ks, st = 3, 2
    p_orig  = torch.randn(B, C, H, W)
    p_comp  = p_orig.unsqueeze(-1).expand(-1, -1, -1, -1, K).contiguous() / K
    
    # Test softmax mode
    alpha   = estimate_alpha(p_orig, ks, st)
    out_softmax = maxpool_decompose(p_comp, p_orig, ks, st, alpha=alpha)
    ref     = F.max_pool2d(p_orig, ks, st)
    err_softmax = (out_softmax.sum(-1) - ref).abs().max().item()
    status  = "✓" if err_softmax < 0.1 else "⚠"
    print(f"maxpool_decompose (softmax)   mass-conservation error: {err_softmax:.2e}  {status}")
    print(f"  estimated α = {alpha:.2f}")
    
    # Test max_pixel mode
    out_max_pixel = maxpool_decompose_max_pixel(p_comp, p_orig, ks, st)
    err_max_pixel = (out_max_pixel.sum(-1) - ref).abs().max().item()
    status = "✓" if err_max_pixel < 0.01 else "⚠"
    print(f"maxpool_decompose (max_pixel) mass-conservation error: {err_max_pixel:.2e}  {status}")


if __name__ == "__main__":
    _run_tests()
