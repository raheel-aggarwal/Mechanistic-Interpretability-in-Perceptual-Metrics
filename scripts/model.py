"""
model.py
--------
AlexNet model utilities: loading, BatchNorm folding, layer geometry, and
the grounding (standard) forward pass that collects activations for the
decomposed pass.

Public API
----------
ALEXNET_CONV_LAYERS  : list of (name, kernel_N, padding, stride)
ALEXNET_POOL_LAYERS  : list of (name, kernel_size, stride, padding)

load_alexnet(pretrained)         -> nn.Module
fold_batch_norm(conv, bn)        -> nn.Conv2d
collect_grounding_values(x, model) -> Dict
"""

from typing import Dict

import torch
import torch.nn as nn
import torchvision.models as tv_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── AlexNet layer geometry ────────────────────────────────────────────────────
# Used by decomposed_forward to look up names and parameters.
# Layer   Type      Kernel  Pad  Stride  Basis-N
# conv1   Conv      11×11    2     4       11
# pool1   MaxPool    3×3     0     2
# conv2   Conv       5×5     2     1        5
# pool2   MaxPool    3×3     0     2
# conv3   Conv       3×3     1     1        3
# conv4   Conv       3×3     1     1        3
# conv5   Conv       3×3     1     1        3

ALEXNET_CONV_LAYERS = [
    # (layer_name, kernel_N, padding, stride)
    ("conv1", 11, 2, 4),
    ("conv2",  5, 2, 1),
    ("conv3",  3, 1, 1),
    ("conv4",  3, 1, 1),
    ("conv5",  3, 1, 1),
]

ALEXNET_POOL_LAYERS = [
    # (layer_name, kernel_size, stride, padding)
    ("pool1", 3, 2, 0),
    ("pool2", 3, 2, 0),
]


def load_alexnet(pretrained: bool = True) -> nn.Module:
    """Load AlexNet (optionally pretrained) and set to eval mode."""
    weights = tv_models.AlexNet_Weights.DEFAULT if pretrained else None
    model   = tv_models.alexnet(weights=weights).to(DEVICE).eval()
    return model


def fold_batch_norm(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fold BatchNorm parameters into the preceding Conv2d.

    BN is affine: y = gamma*(x - mu)/sigma + beta.
    Folding gives an equivalent conv with:
        W' = (gamma/sigma) * W
        b' = (gamma/sigma) * (b - mu) + beta

    Required for exact basis decomposition (keeps the network linear).
    Returns a new Conv2d; original layers are unchanged.
    """
    with torch.no_grad():
        gamma  = bn.weight
        beta   = bn.bias
        mu     = bn.running_mean
        sigma  = torch.sqrt(bn.running_var + bn.eps)
        scale  = gamma / sigma                         # (Co,)

        w_new  = conv.weight * scale.view(-1, 1, 1, 1)
        b_orig = conv.bias if conv.bias is not None else torch.zeros_like(mu)
        b_new  = scale * (b_orig - mu) + beta

        folded = nn.Conv2d(
            conv.in_channels, conv.out_channels,
            conv.kernel_size, stride=conv.stride,
            padding=conv.padding, bias=True,
        ).to(conv.weight.device)
        folded.weight.data.copy_(w_new)
        folded.bias.data.copy_(b_new)
    return folded


def collect_grounding_values(
    image_tensor: torch.Tensor,   # (1, 3, H, W)
    model:        nn.Module,
) -> Dict:
    """
    Standard (undecomposed) forward pass through AlexNet features.

    Walks model.features layer by layer and records:
      - conv_outputs[i] : post-conv pre-ReLU activations  (Z)
      - relu_gates[i]   : 1[Z > 0] boolean tensor for i-th ReLU
      - relu_outputs[i] : post-ReLU activations
      - pool_inputs[i]  : pre-MaxPool activations
      - final_features  : output after the last MaxPool

    AlexNet features index reference:
      0:Conv1  1:ReLU  2:MaxPool
      3:Conv2  4:ReLU  5:MaxPool
      6:Conv3  7:ReLU
      8:Conv4  9:ReLU
      10:Conv5 11:ReLU 12:MaxPool

    Returns
    -------
    dict with keys:
        'conv_outputs', 'relu_gates', 'relu_outputs',
        'pool_inputs',  'final_features'
    """
    grounding: Dict = {
        "conv_outputs": [],
        "relu_gates":   [],
        "relu_outputs": [],
        "pool_inputs":  [],
    }

    x = image_tensor.to(DEVICE)
    with torch.no_grad():
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                grounding["conv_outputs"].append(x.clone().cpu())
                grounding["relu_gates"].append((x > 0).cpu())
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
                grounding["relu_outputs"].append(x.clone().cpu())
            elif isinstance(layer, nn.MaxPool2d):
                grounding["pool_inputs"].append(x.clone().cpu())
                x = layer(x)
            else:
                x = layer(x)

    grounding["final_features"] = x.clone().cpu()
    return grounding
