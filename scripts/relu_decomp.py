"""
relu_decomp.py
--------------
Exact ReLU decomposition via grounding.

Public API
----------
relu_decompose(z_components, Z_original) -> Tensor (..., K)
"""

import torch


def relu_decompose(
    z_components: torch.Tensor,
    Z_original:   torch.Tensor,
) -> torch.Tensor:
    """
    Exact ReLU decomposition via grounding.

    For each output neuron j with pre-activation Z_j (from the grounding pass):
        f(z_{ij})  =  z_{ij} * 1[Z_j > 0]

    Correctness: sum_i z_{ij} = Z_j exactly (linear basis expansion), so
        sum_i f(z_{ij}) = 1[Z_j > 0] * sum_i z_{ij} = ReLU(Z_j)   ✓

    The gate uses strict inequality (Z_j > 0), consistent with PyTorch default.

    Parameters
    ----------
    z_components : Tensor  (..., K)
        Basis-component activations at this layer.
    Z_original   : Tensor  (...)
        Pre-ReLU activations from the grounding forward pass.
        Shape must broadcast with z_components[..., 0].

    Returns
    -------
    Tensor  (..., K)  — post-ReLU component activations.
    """
    gate = (Z_original > 0).to(z_components.dtype)   # 1[Z_j > 0], strict >
    return z_components * gate.unsqueeze(-1)


# ── Self-test ─────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    z_comp = torch.tensor([[1.0, -2.0, 3.0],
                            [0.5,  0.5, -1.0]])
    Z_orig = torch.tensor([2.0, -1.0])
    out    = relu_decompose(z_comp, Z_orig)

    assert torch.allclose(out[0], z_comp[0]),       "Active neuron should pass through unchanged"
    assert torch.allclose(out[1], torch.zeros(3)),  "Inactive neuron should be zeroed"
    print("relu_decompose  ✓")
    print(f"  active   neuron: {out[0].tolist()}")
    print(f"  inactive neuron: {out[1].tolist()}")


if __name__ == "__main__":
    _run_tests()
