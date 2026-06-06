"""
attribution.py
--------------
Post-processing: component attribution summaries, reconstruction error,
and image preprocessing.

Public API
----------
preprocess_image(image_path, resize, crop) -> Tensor (1, 3, H, W)
compute_attributions(layer_components, target_layer, ...) -> Dict
attribution_to_pq(flat_idx, N)             -> (p, q)
compute_decomposition_error(decomp_results, grounding, ...) -> float
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def preprocess_image(
    image_path: str,
    resize:     int = 256,
    crop:       int = 224,
) -> torch.Tensor:
    """
    Standard ImageNet preprocessing.

    Returns
    -------
    Tensor  (1, 3, crop, crop)
    """
    tfm = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return tfm(img).unsqueeze(0)


def compute_attributions(
    layer_components: Dict[str, torch.Tensor],
    target_layer:     str,
    path_epsilon:     float          = 1e-6,
    top_k:            Optional[int]  = 20,
) -> Dict:
    """
    Summarise basis-component attributions for one layer.

    For each output channel: compute the mean |z| per basis component over
    all spatial positions and the batch dimension.

    Parameters
    ----------
    layer_components : {layer_name: Tensor (B,C,H,W,K)}
    target_layer     : which layer to summarise
    path_epsilon     : zero out components below this threshold
    top_k            : return only the top-k components per channel

    Returns
    -------
    dict with:
      'channel_attributions' : ndarray (C, K) — mean |z|
      'top_k_indices'        : ndarray (C, top_k)
      'top_k_values'         : ndarray (C, top_k)
      'basis_N'              : int   (sqrt of K)
      'layer'                : str
    """
    comp = layer_components[target_layer]   # (B, C, H, W, K)
    B, C, H, W, K = comp.shape

    mean_abs = comp.abs().mean(dim=(0, 2, 3)).numpy()   # (C, K)
    mean_abs[mean_abs < path_epsilon] = 0.0

    N   = int(math.isqrt(K))
    out = {
        "channel_attributions": mean_abs,
        "basis_N": N,
        "layer":   target_layer,
    }

    if top_k is not None:
        tk   = min(top_k, K)
        idx  = np.argsort(mean_abs, axis=-1)[:, ::-1][:, :tk]
        vals = np.take_along_axis(mean_abs, idx, axis=-1)
        out["top_k_indices"] = idx
        out["top_k_values"]  = vals

    return out


def attribution_to_pq(flat_idx: int, N: int) -> Tuple[int, int]:
    """Convert flat basis index to (p, q) in row-major order."""
    return divmod(flat_idx, N)


def compute_decomposition_error(
    decomp_results: Dict,
    grounding:      Dict,
    layer:          str = "conv5",
    conv_index:     int = 4,
) -> float:
    """
    Normalised L2 error between summed decomposition components and the CNN
    activation at the same point in the network.

    layer_components[layer] holds POST-CONV PRE-RELU components.  We apply
    the same ReLU gate the decomposed forward uses next, then sum — giving
    the POST-RELU decomposed reconstruction to compare against
    grounding["relu_outputs"][conv_index].

    Normalised L2  =  ||decomp_relu_sum - cnn_ref||₂  /  ||cnn_ref||₂

    Parameters
    ----------
    layer       : layer name in layer_components  (default "conv5")
    conv_index  : 0-based conv index used to look up
                  conv_outputs[conv_index]  (ReLU gate)  and
                  relu_outputs[conv_index]  (reference)
                  For conv5 → index 4.

    Returns
    -------
    float — normalised L2 distance (0 = perfect reconstruction)
    """
    comp = decomp_results["layer_components"].get(layer)
    if comp is None:
        raise KeyError(f"Layer {layer!r} not found in decomp_results.")

    conv_outs = grounding.get("conv_outputs", [])
    if conv_index >= len(conv_outs):
        raise IndexError(
            f"conv_index={conv_index} out of range "
            f"({len(conv_outs)} conv outputs stored).")

    Z_orig     = conv_outs[conv_index].float()
    gate       = (Z_orig > 0).to(comp.dtype)
    comp_relu  = comp.float() * gate.unsqueeze(-1)
    decomp_sum = comp_relu.sum(-1)

    relu_outs = grounding.get("relu_outputs", [])
    if conv_index >= len(relu_outs):
        raise IndexError(
            f"conv_index={conv_index} out of range "
            f"({len(relu_outs)} relu outputs stored).")
    cnn_ref = relu_outs[conv_index].float()

    diff_norm = torch.norm(decomp_sum - cnn_ref).item()
    ref_norm  = torch.norm(cnn_ref).item()
    if ref_norm < 1e-12:
        return float("inf")
    return diff_norm / ref_norm
