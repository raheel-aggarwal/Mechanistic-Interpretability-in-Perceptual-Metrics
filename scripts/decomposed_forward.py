"""
decomposed_forward.py
---------------------
Pass 2 of the pipeline: propagate basis components through AlexNet.

Public API
----------
decomposed_forward(image_tensor, model, grounding, ...) -> Dict
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from basis       import haar_basis, dct_basis
from kernel_decomp import basis_component_maps, prune_components
from relu_decomp   import relu_decompose
from maxpool_decomp import maxpool_decompose, estimate_alpha
from model         import ALEXNET_CONV_LAYERS, ALEXNET_POOL_LAYERS, DEVICE


def decomposed_forward(
    image_tensor:       torch.Tensor,
    model:              nn.Module,
    grounding:          Dict,
    basis_type:         str            = "haar",
    alpha_per_pool:     Dict           = None,
    alpha_auto_target:  float          = 30.0,
    alpha_max:          float          = 200.0,
    alpha_fallback:     float          = 50.0,
    path_epsilon:       float          = 1e-6,
    prune:              Optional[int]  = None,
) -> Dict:
    """
    Decomposed forward pass through AlexNet features.

    Each conv layer output is represented as a Tensor (B, C_out, H, W, K)
    where K = N² for basis decompositions or K = 1 for basis_type="none".
    ReLU and MaxPool stages propagate the component tensor exactly (ReLU)
    or via softmax approximation (MaxPool).

    Parameters
    ----------
    image_tensor    : Tensor (1, 3, H, W)  — preprocessed input image
    model           : AlexNet in eval mode
    grounding       : dict from collect_grounding_values()
    basis_type      : "haar" | "dct" | "none"
        "haar" / "dct" — decompose each conv kernel into N² basis components.
        "none"         — K=1; single component = plain conv output from
                         grounding (exact, no basis math, all downstream
                         stages unchanged).
    alpha_per_pool  : {pool_index: float | None}  — manual α per MaxPool;
                      None means auto-select via estimate_alpha.
    alpha_auto_target : target α * gap for auto α selection
    alpha_max       : hard α ceiling (overflow guard)
    alpha_fallback  : α used when gap estimation fails (near-tie inputs)
    path_epsilon    : unused here; passed through to results for attribution
    prune           : int or None — keep only top-`prune` components per
                      channel after each conv.  With basis_type="none" K=1
                      so pruning has no effect.

    Returns
    -------
    dict with:
      'layer_components' : {layer_name: Tensor (B,C,H,W,K)} — CPU tensors
                            K = N² for haar/dct, K = 1 for none
      'per_layer_alpha'  : {pool_name: float}
      'basis_type'       : str
      'prune'            : int or None
    """
    if alpha_per_pool is None:
        alpha_per_pool = {}

    feat       = model.features
    x          = image_tensor.to(DEVICE)
    x_comp     = None   # (B, C, H, W, K) — initialised at first conv
    results    = {"layer_components": {}, "per_layer_alpha": {}}

    relu_idx  = 0
    pool_idx  = 0
    conv_idx  = 0

    conv_names = [n for n, _, _, _ in ALEXNET_CONV_LAYERS]
    pool_names = [n for n, _, _, _ in ALEXNET_POOL_LAYERS]

    with torch.no_grad():
        for layer in feat:

            # ── Conv ──────────────────────────────────────────────────────
            if isinstance(layer, nn.Conv2d):
                N_kern   = layer.kernel_size[0]
                pad      = layer.padding[0]
                K_weight = layer.weight
                lname    = (conv_names[conv_idx]
                            if conv_idx < len(conv_names)
                            else f"conv{conv_idx}")

                if basis_type == "none":
                    # K=1: plain conv output from grounding, no basis math.
                    # Shape: (B, Co, Ho, Wo, 1)
                    z_pq = (grounding["conv_outputs"][conv_idx]
                            .to(DEVICE)
                            .unsqueeze(-1))

                else:
                    U = (haar_basis(N_kern) if basis_type == "haar"
                         else dct_basis(N_kern))

                    if x_comp is None:
                        # First conv: image has no component tensor yet
                        z_pq = basis_component_maps(
                            x, K_weight, U,
                            padding=pad, stride=layer.stride[0])
                    else:
                        # Subsequent convs: accumulate over previous K components
                        B, Ci, H, W, K_prev = x_comp.shape
                        z_acc = torch.zeros_like(
                            basis_component_maps(
                                x_comp[..., 0], K_weight, U,
                                padding=pad, stride=layer.stride[0]))
                        for k in range(K_prev):
                            z_acc = z_acc + basis_component_maps(
                                x_comp[..., k], K_weight, U,
                                padding=pad, stride=layer.stride[0])
                        z_pq = z_acc

                x_comp = z_pq

                # Optional per-channel pruning
                if prune is not None:
                    x_comp = prune_components(x_comp, prune)

                results["layer_components"][lname] = x_comp.clone().cpu()

                K_now    = x_comp.shape[-1]
                n_active = int((x_comp.abs().sum(dim=(0, 1, 2, 3)) > 0)
                               .sum().item())
                if prune is not None:
                    print(f"    {lname}: {K_now} components "
                          f"→ {n_active} active after pruning")
                else:
                    print(f"    {lname}: {K_now} components")

                conv_idx += 1
                x = layer(x)   # advance grounded reference

            # ── ReLU ──────────────────────────────────────────────────────
            elif isinstance(layer, nn.ReLU):
                if x_comp is not None:
                    Z_orig = grounding["conv_outputs"][relu_idx].to(DEVICE)
                    x_comp = relu_decompose(x_comp, Z_orig)
                relu_idx += 1
                x = layer(x)

            # ── MaxPool ───────────────────────────────────────────────────
            elif isinstance(layer, nn.MaxPool2d):
                ks     = (layer.kernel_size if isinstance(layer.kernel_size, int)
                          else layer.kernel_size[0])
                st     = (layer.stride  if isinstance(layer.stride, int)
                          else layer.stride[0])
                pad_mp = (layer.padding if isinstance(layer.padding, int)
                          else layer.padding[0])
                pname  = (pool_names[pool_idx]
                          if pool_idx < len(pool_names)
                          else f"pool{pool_idx}")

                p_orig = grounding["pool_inputs"][pool_idx].to(DEVICE)

                if alpha_per_pool.get(pool_idx) is not None:
                    alpha_val = float(alpha_per_pool[pool_idx])
                else:
                    alpha_val = estimate_alpha(
                        p_orig, ks, st,
                        target_gap_product = alpha_auto_target,
                        alpha_max          = alpha_max,
                        alpha_fallback     = alpha_fallback,
                    )

                results["per_layer_alpha"][pname] = alpha_val

                if x_comp is not None:
                    x_comp = maxpool_decompose(
                        x_comp, p_orig, ks, st,
                        padding=pad_mp, alpha=alpha_val)

                pool_idx += 1
                x = layer(x)

            else:
                x = layer(x)

    results["basis_type"] = basis_type
    results["prune"]      = prune
    return results
