"""
pipeline.py
-----------
High-level pipeline orchestration for AlexNet basis decomposition.

This module contains model loading, grounding, attribution summary, output
serialization, visualisation, and the decomposed forward pass that stitches
kernel, ReLU and MaxPool decomposition modules together.
"""

import csv
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from kernel_decomp import (
    basis_component_maps,
    dct_basis,
    haar_basis,
    project_kernel,
    prune_components,
    reconstruct_from_coeffs,
)
from maxpool_decomp import estimate_alpha, maxpool_decompose, maxpool_decompose_max_pixel
from relu_decomp import relu_decompose

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALEXNET_CONV_LAYERS = [
    ("conv1", 11, 2, 4),
    ("conv2", 5, 2, 1),
    ("conv3", 3, 1, 1),
    ("conv4", 3, 1, 1),
    ("conv5", 3, 1, 1),
]

ALEXNET_POOL_LAYERS = [
    ("pool1", 3, 2, 0),
    ("pool2", 3, 2, 0),
]


def load_basis_json(path: Path) -> Dict:
    """Load a basis JSON specification from `path`.

    The JSON may either contain explicit numeric vectors under keys `1d`/`2d` or
    a `{ "generate": "haar" }` / `{ "generate": "dct" }` instruction for
    each kernel size. This function returns the parsed dict unchanged; the
    decomposed forward pass will resolve numeric arrays as needed.
    """
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def _resolve_1d_basis(basis_spec, N: int):
    """Return an (N,N) ndarray of 1-D basis vectors for size N.

    `basis_spec` may be:
    - the string 'haar' or 'dct' → compute with the builtin generator
    - the dict loaded from a basis JSON → it may contain `1d` entries keyed by
      the kernel size (as strings). Each entry may either be a list-of-lists
      (explicit vectors) or an object {"generate": "haar"|"dct"}.
    """
    if basis_spec is None or basis_spec == "none":
        return None
    if isinstance(basis_spec, str):
        if basis_spec == "haar":
            return haar_basis(N)
        if basis_spec == "dct":
            return dct_basis(N)
        # unexpected string, treat as none
        return None

    # assume dict-like JSON spec
    # Expect a `dim` attribute: 1 (1D vectors) or 2 (explicit 2D kernels).
    if isinstance(basis_spec, dict):
        if basis_spec.get("dim") == 2:
            raise ValueError(
                "Provided basis JSON contains 2-D kernels; a 1-D basis is required here"
            )

        one_d = basis_spec.get("1d", {})
        key = str(N)
        entry = one_d.get(key) or one_d.get(N)
        if entry is None:
            # fallback to declared basis flavour if present
            flavour = basis_spec.get("basis")
            if flavour == "dct":
                return dct_basis(N)
            return haar_basis(N)

        # only numeric list-of-lists is accepted; 'generate' instruction removed
        if not isinstance(entry, list):
            raise ValueError(
                "basis JSON entries must be numeric arrays under '1d' for each kernel size"
            )
        return np.array(entry)

    # unexpected type — treat as none
    return None


def load_alexnet(pretrained: bool = True) -> nn.Module:
    """Load AlexNet and move it to the active device."""
    weights = tv_models.AlexNet_Weights.DEFAULT if pretrained else None
    model = tv_models.alexnet(weights=weights).to(DEVICE).eval()
    return model


def collect_grounding_values(
    image_tensor: torch.Tensor,
    model:        nn.Module,
) -> Dict:
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


def preprocess_image(
    image_path: str,
    resize:     int = 256,
    crop:       int = 224,
) -> torch.Tensor:
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
    comp = layer_components[target_layer]
    B, C, H, W, K = comp.shape

    mean_abs = comp.abs().mean(dim=(0, 2, 3)).numpy()
    mean_abs[mean_abs < path_epsilon] = 0.0

    out = {
        "channel_attributions": mean_abs,
        "basis_N": int(math.isqrt(K)),
        "layer": target_layer,
    }

    if top_k is not None:
        top_k = min(top_k, K)
        idx = np.argsort(mean_abs, axis=-1)[:, ::-1][:, :top_k]
        out["top_k_indices"] = idx
        out["top_k_values"]  = np.take_along_axis(mean_abs, idx, axis=-1)

    return out


def attribution_to_pq(flat_idx: int, N: int) -> Tuple[int, int]:
    return divmod(flat_idx, N)


def compute_reconstruction_error(
    decomp_sum: torch.Tensor,
    reference:  torch.Tensor,
) -> float:
    reference = reference.to(decomp_sum.dtype).to(decomp_sum.device)
    diff_norm = torch.norm(decomp_sum - reference).item()
    ref_norm  = torch.norm(reference).item()
    return float("inf") if ref_norm < 1e-12 else diff_norm / ref_norm


def compute_decomposition_error(
    decomp_results: Dict,
    grounding:      Dict,
    layer:          str = "conv5",
    conv_index:     int = 4,
) -> float:
    comp = decomp_results["layer_components"].get(layer)
    if comp is None:
        raise KeyError(f"Layer {layer!r} not found in decomp_results.")

    Z_orig = grounding["conv_outputs"][conv_index].float()
    gate   = (Z_orig > 0).to(comp.dtype)
    comp_relu = comp.float() * gate.unsqueeze(-1)
    decomp_sum = comp_relu.sum(-1)

    cnn_ref = grounding["relu_outputs"][conv_index].float()
    return compute_reconstruction_error(decomp_sum, cnn_ref)


def save_outputs(
    grounding:      Dict,
    decomp_results: Dict,
    attributions:   Dict,
    output_dir:     Path,
    cfg:            Dict = None,
) -> None:
    cfg = cfg or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "grounding.pkl", "wb") as f:
        pickle.dump(grounding, f, protocol=pickle.HIGHEST_PROTOCOL)

    comp_np = {
        k: v.numpy() if isinstance(v, torch.Tensor) else v
        for k, v in decomp_results["layer_components"].items()
    }
    meta = {k: v for k, v in decomp_results.items() if k != "layer_components"}
    with open(output_dir / "components.pkl", "wb") as f:
        pickle.dump({"components": comp_np, "meta": meta}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    def _to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_json(v) for k, v in obj.items()}
        return obj

    with open(output_dir / "attributions.json", "w") as f:
        json.dump(_to_json(attributions), f, indent=2)

    print(f"Saved outputs to {output_dir.resolve()}")


def load_outputs(
    output_dir: Path,
    cfg:        Dict = None,
) -> Tuple[Dict, Dict, Dict]:
    cfg = cfg or {}
    output_dir = Path(output_dir)

    with open(output_dir / "grounding.pkl", "rb") as f:
        grounding = pickle.load(f)

    with open(output_dir / "components.pkl", "rb") as f:
        saved = pickle.load(f)

    layer_components = {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        for k, v in saved["components"].items()
    }
    decomp_results = saved["meta"]
    decomp_results["layer_components"] = layer_components

    with open(output_dir / "attributions.json", "r") as f:
        attributions = json.load(f)

    return grounding, decomp_results, attributions


def interp_label(p: int, q: int, basis_type: str) -> str:
    if basis_type == "none":
        return "(raw)"

    def _axis(v: int, bt: str) -> str:
        if bt == "haar":
            return "mean" if v == 0 else f"s{v}"
        return "DC" if v == 0 else f"f{v}"

    return f"({_axis(p, basis_type)},{_axis(q, basis_type)})"


def get_smap(dr: Dict, layer: str, channel: int, flat_idx: int) -> np.ndarray:
    return dr["layer_components"][layer][0, channel, :, :, flat_idx].numpy()


def energy_pct(smap_k: np.ndarray, smap_total: np.ndarray) -> float:
    total_e = float(np.sum(smap_total ** 2))
    if total_e < 1e-12:
        return 0.0
    return 100.0 * float(np.sum(smap_k ** 2)) / total_e


def top_flat_indices(
    dr:    Dict,
    layer: str,
    channel: int,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    comp   = dr["layer_components"][layer]
    mean_a = comp[0, channel].abs().mean(dim=(0, 1)).numpy()
    idx    = np.argsort(mean_a)[::-1][:top_k]
    return idx, mean_a


def total_smap(dr: Dict, layer: str, channel: int) -> np.ndarray:
    return dr["layer_components"][layer][0, channel, :, :, :].sum(-1).numpy()


def denorm_image(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img  = np.clip(img * std + mean, 0, 1)
    return (img * 255).astype(np.uint8)


def basis_kernel_rgb(U: np.ndarray, p: int, q: int) -> np.ndarray:
    psi = np.outer(U[p], U[q])
    lo, hi = psi.min(), psi.max()
    psi_n = (psi - lo) / max(hi - lo, 1e-8)
    return (np.stack([psi_n] * 3, axis=-1) * 255).astype(np.uint8)


def write_stats_csvs(
    dr:         Dict,
    output_dir: Path,
    basis_type: str,
    top_k:      int = 20,
) -> None:
    stats_dir = Path(output_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    for lname, comp in dr["layer_components"].items():
        B, C, H, W, K = comp.shape
        N = int(math.isqrt(K))

        csv_path = stats_dir / f"{lname}_top{top_k}_stats.csv"
        with open(csv_path, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["channel", "rank", "flat_idx", "p", "q",
                             "mean_abs_z", "energy_pct_of_channel", "interp"])
            for ch in range(C):
                idx, mean_a = top_flat_indices(dr, lname, ch, top_k)
                smap_tot = total_smap(dr, lname, ch)
                total_e = float(np.sum(smap_tot ** 2)) + 1e-12
                for rank, flat in enumerate(idx, 1):
                    p, q = attribution_to_pq(int(flat), N)
                    smap_k = get_smap(dr, lname, ch, int(flat))
                    e_pct = 100.0 * float(np.sum(smap_k ** 2)) / total_e
                    writer.writerow([ch, rank, int(flat), p, q,
                                     f"{mean_a[flat]:.6f}",
                                     f"{e_pct:.4f}",
                                     interp_label(p, q, basis_type)])
        print(f"  {lname}: stats → {csv_path.name}")

    print(f"  CSV stats written to {stats_dir.resolve()}")


def write_attribution_heatmaps(
    dr:         Dict,
    output_dir: Path,
    basis_type: str,
) -> None:
    hm_dir = Path(output_dir) / "heatmaps"
    hm_dir.mkdir(parents=True, exist_ok=True)

    for lname, comp in dr["layer_components"].items():
        B, C, H, W, K = comp.shape
        N = int(math.isqrt(K))
        data = comp[0].abs().mean(dim=(1, 2)).numpy()

        fig = plt.figure(figsize=(max(14, K // 3), max(5, C // 8 + 2)))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.35)

        ax0 = fig.add_subplot(gs[0])
        im = ax0.imshow(data, aspect="auto", interpolation="nearest",
                        norm=mcolors.PowerNorm(gamma=0.4), cmap="viridis")
        ax0.set_xlabel(f"Basis component (flat index, N={N})", fontsize=9)
        ax0.set_ylabel("Output channel", fontsize=9)
        ax0.set_title(f"Attribution heatmap — {lname}  [{basis_type}]", fontsize=10)
        ax0.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        plt.colorbar(im, ax=ax0, label="Mean |z|", shrink=0.8)

        ax1 = fig.add_subplot(gs[1])
        marginal = data.mean(axis=0)
        ax1.barh(np.arange(K), marginal[::-1], color="steelblue", height=0.8)
        tick_step = max(1, K // 8)
        ax1.set_yticks(np.arange(0, K, tick_step))
        ax1.set_yticklabels([str(K - 1 - i) for i in range(0, K, tick_step)], fontsize=7)
        ax1.set_xlabel("Mean |z| (avg over channels)", fontsize=8)
        ax1.set_title("Component marginal", fontsize=9)
        ax1.invert_yaxis()

        out_path = hm_dir / f"{lname}_attribution_heatmap.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {lname} heatmap → {out_path.name}")

    print(f"  Heatmaps written to {hm_dir.resolve()}")


def write_dominant_maps(
    dr:           Dict,
    output_dir:   Path,
    basis_type:   str,
    max_channels: int = 96,
) -> None:
    dom_dir = Path(output_dir) / "dominant_maps"
    dom_dir.mkdir(parents=True, exist_ok=True)

    for lname, comp in dr["layer_components"].items():
        B, C, H, W, K = comp.shape
        n_shown  = min(C, max_channels)
        n_cols   = min(12, n_shown)
        n_rows   = math.ceil(n_shown / n_cols)
        cell_px  = max(1.8, 110 / max(H, W))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * cell_px * 1.05, n_rows * (cell_px + 0.55)))
        axes_flat = np.array(axes).flatten()

        for ch in range(n_shown):
            ax = axes_flat[ch]
            idx, _ = top_flat_indices(dr, lname, ch, top_k=1)
            flat = idx[0]
            p, q = attribution_to_pq(int(flat), int(math.isqrt(K)))
            smap = get_smap(dr, lname, ch, int(flat))
            tot  = total_smap(dr, lname, ch)
            e_pct = energy_pct(smap, tot)
            vmax = max(np.abs(smap).max(), 1e-8)
            ax.imshow(smap, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
            ax.set_title(
                f"ch{ch}\n{interp_label(p, q, basis_type)}\n{e_pct:.1f}%",
                fontsize=6, pad=2)
            ax.axis("off")

        for ax in axes_flat[n_shown:]:
            ax.axis("off")

        fig.suptitle(
            f"Dominant basis component per channel — {lname}  [{basis_type}]",
            fontsize=10, y=1.01)
        plt.tight_layout(pad=0.4)

        out_path = dom_dir / f"{lname}_dominant_component_maps.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {lname}: {n_shown} dominant maps → {out_path.name}")

    print(f"  Dominant maps written to {dom_dir.resolve()}")


def write_topk_panels(
    dr:           Dict,
    output_dir:   Path,
    basis_type:   str,
    top_k:        int = 20,
    max_channels: int = 96,
) -> None:
    topk_dir = Path(output_dir) / "topk_maps"

    for lname, comp in dr["layer_components"].items():
        B, C, H, W, K = comp.shape
        N = int(math.isqrt(K))
        Kd = min(top_k, K)
        layer_dir = topk_dir / lname
        layer_dir.mkdir(parents=True, exist_ok=True)

        n_ch = min(C, max_channels)
        for ch in range(n_ch):
            idx, mean_a = top_flat_indices(dr, lname, ch, Kd)
            smap_tot = total_smap(dr, lname, ch)
            total_e = float(np.sum(smap_tot ** 2)) + 1e-12
            total_shown_e = sum(
                float(np.sum(get_smap(dr, lname, ch, int(f)) ** 2))
                for f in idx) / total_e * 100

            fig = plt.figure(figsize=(7, max(2, Kd * 0.85 + 0.6)))
            outer_gs = gridspec.GridSpec(Kd, 3, figure=fig,
                                         width_ratios=[0.9, 3, 1.5],
                                         hspace=0.15, wspace=0.08)

            for rank, flat in enumerate(idx):
                p, q   = attribution_to_pq(int(flat), N)
                smap_k = get_smap(dr, lname, ch, int(flat))
                e_pct  = 100.0 * float(np.sum(smap_k ** 2)) / total_e

                ax_lbl = fig.add_subplot(outer_gs[rank, 0])
                ax_lbl.text(0.5, 0.5,
                            f"#{rank+1}\n{interp_label(p, q, basis_type)}\nflat={flat}",
                            ha="center", va="center", fontsize=7,
                            transform=ax_lbl.transAxes)
                ax_lbl.axis("off")

                ax_map = fig.add_subplot(outer_gs[rank, 1])
                vmax = max(np.abs(smap_k).max(), 1e-8)
                ax_map.imshow(smap_k, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                              interpolation="nearest", aspect="equal")
                ax_map.axis("off")

                ax_bar = fig.add_subplot(outer_gs[rank, 2])
                bar_col = plt.cm.YlOrRd(min(e_pct / 50, 1.0))
                ax_bar.barh([0], [e_pct], color=bar_col, height=0.6)
                ax_bar.set_xlim(0, max(total_shown_e * 1.05, 1))
                ax_bar.set_ylim(-0.5, 0.5)
                ax_bar.set_yticks([])
                ax_bar.tick_params(labelsize=6)
                ax_bar.set_xlabel("Energy %", fontsize=6)
                ax_bar.text(e_pct + total_shown_e * 0.02, 0,
                            f"{e_pct:.1f}%", va="center", fontsize=6)
                if rank == 0:
                    ax_bar.set_title("Energy %", fontsize=7)

            fig.suptitle(
                f"{lname}  ch={ch}  |  top-{Kd}  [{basis_type}]  "
                f"|  cumulative energy: {total_shown_e:.1f}%",
                fontsize=8, y=1.005)
            out_path = layer_dir / f"ch{ch:04d}_top{Kd}_components.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

        print(f"  {lname}: {n_ch} channel panels → {layer_dir.name}/")

    print(f"  Top-K panels written to {topk_dir.resolve()}")


def write_conv1_rgb_panels(
    dr:          Dict,
    output_dir:  Path,
    basis_type:  str,
    image_path:  str,
    top_k:       int = 20,
) -> None:
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset
    except ImportError:
        _inset = None

    rgb_dir = Path(output_dir) / "conv1_rgb_panels"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    x_tensor = preprocess_image(image_path)
    img_rgb = denorm_image(x_tensor)

    comp1 = dr["layer_components"]["conv1"]
    B, C, Ho, Wo, K = comp1.shape
    N1 = int(math.isqrt(K))
    U1 = haar_basis(N1) if basis_type == "haar" else dct_basis(N1)
    Kd = min(top_k, K)
    H_in, W_in = 224, 224

    for ch in range(C):
        idx, _ = top_flat_indices(dr, "conv1", ch, Kd)
        smap_tot = total_smap(dr, "conv1", ch)
        total_e = float(np.sum(smap_tot ** 2)) + 1e-12

        n_rows = Kd + 1
        fig = plt.figure(figsize=(11, n_rows * 1.45 + 0.8))
        gs_main = gridspec.GridSpec(n_rows, 4, figure=fig,
                                    width_ratios=[1.1, 2.8, 2.0, 1.5],
                                    hspace=0.08, wspace=0.06)

        ax_lbl = fig.add_subplot(gs_main[0, 0])
        ax_lbl.text(0.5, 0.5, f"ch {ch}\nOriginal",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    transform=ax_lbl.transAxes)
        ax_lbl.axis("off")

        ax_img = fig.add_subplot(gs_main[0, 1])
        ax_img.imshow(img_rgb)
        ax_img.set_title("Original image (224×224)", fontsize=8)
        ax_img.axis("off")

        ax_tot = fig.add_subplot(gs_main[0, 2])
        vmax_t = max(np.abs(smap_tot).max(), 1e-8)
        ax_tot.imshow(smap_tot, cmap="RdBu_r", vmin=-vmax_t, vmax=vmax_t,
                      interpolation="nearest")
        ax_tot.set_title(f"Conv1 ch{ch} total output (55×55)", fontsize=8)
        ax_tot.axis("off")
        fig.add_subplot(gs_main[0, 3]).axis("off")

        top1_e = 100.0 * float(np.sum(get_smap(dr, "conv1", ch, int(idx[0])) ** 2)) / total_e

        for rank, flat in enumerate(idx):
            row = rank + 1
            p, q = attribution_to_pq(int(flat), N1)
            smap_k = get_smap(dr, "conv1", ch, int(flat))
            e_pct  = 100.0 * float(np.sum(smap_k ** 2)) / total_e
            psi_vis = basis_kernel_rgb(U1, p, q)

            smap_t = torch.tensor(smap_k).unsqueeze(0).unsqueeze(0).float()
            smap_up = F.interpolate(smap_t, size=(H_in, W_in), mode="bilinear", align_corners=False)
            smap_up = smap_up.squeeze().numpy()

            vmax_up = max(np.abs(smap_up).max(), 1e-8)
            alpha = np.clip((np.abs(smap_up) / vmax_up) * 0.65, 0, 0.65)
            sgn = np.sign(smap_up)
            r_ch = np.clip(img_rgb[:, :, 0] / 255. + alpha * np.clip(sgn, 0, 1) * 0.6, 0, 1)
            b_ch = np.clip(img_rgb[:, :, 2] / 255. + alpha * np.clip(-sgn, 0, 1) * 0.6, 0, 1)
            g_ch = np.clip(img_rgb[:, :, 1] / 255. * (1 - alpha * 0.4), 0, 1)
            composite = np.stack([r_ch, g_ch, b_ch], axis=-1)

            ax_lbl = fig.add_subplot(gs_main[row, 0])
            ax_lbl.text(0.5, 0.5,
                        f"#{rank+1}\n{interp_label(p, q, basis_type)}\nflat={flat}",
                        ha="center", va="center", fontsize=7,
                        transform=ax_lbl.transAxes)
            ax_lbl.axis("off")

            ax_comp = fig.add_subplot(gs_main[row, 1])
            ax_comp.imshow(composite, interpolation="bilinear")
            if _inset is not None:
                axins = _inset(ax_comp, width="22%", height="22%", loc="upper right")
                axins.imshow(psi_vis, interpolation="nearest")
                axins.axis("off")
            ax_comp.axis("off")
            if rank == 0:
                ax_comp.set_title(
                    "Image × component  (red=+, blue=−)  |  inset: basis kernel",
                    fontsize=7)

            ax_map = fig.add_subplot(gs_main[row, 2])
            vmax_k = max(np.abs(smap_k).max(), 1e-8)
            ax_map.imshow(smap_k, cmap="RdBu_r", vmin=-vmax_k, vmax=vmax_k,
                          interpolation="nearest")
            ax_map.axis("off")
            if rank == 0:
                ax_map.set_title("Component map (55×55)", fontsize=7)

            ax_bar = fig.add_subplot(gs_main[row, 3])
            bar_col = plt.cm.YlOrRd(min(e_pct / 40, 1.0))
            ax_bar.barh([0], [e_pct], color=bar_col, height=0.55)
            ax_bar.set_xlim(0, max(top1_e * 1.15, 1))
            ax_bar.set_ylim(-0.5, 0.5)
            ax_bar.set_yticks([])
            ax_bar.tick_params(labelsize=6)
            ax_bar.text(e_pct + top1_e * 0.03, 0,
                        f"{e_pct:.1f}%", va="center", fontsize=6)
            if rank == 0:
                ax_bar.set_title("Energy %", fontsize=7)

        total_shown_e = sum(
            100.0 * float(np.sum(get_smap(dr, "conv1", ch, int(f)) ** 2)) / total_e
            for f in idx)
        fig.suptitle(
            f"Conv1  ch={ch}  |  top-{Kd} RGB decomposition  [{basis_type}]  "
            f"|  cumulative energy: {total_shown_e:.1f}%",
            fontsize=9, y=1.002)

        out_path = rgb_dir / f"ch{ch:04d}_conv1_rgb_top{Kd}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Conv1 RGB panels ({C} channels) → {rgb_dir.resolve()}")


def write_summary_scatter(
    dr:         Dict,
    output_dir: Path,
    basis_type: str,
) -> None:
    n_layers = len(dr["layer_components"])
    fig, axes = plt.subplots(
        1, n_layers,
        figsize=(max(5 * n_layers, 8), 5),
        squeeze=False)

    for col, (lname, comp) in enumerate(dr["layer_components"].items()):
        B, C, H, W, K = comp.shape
        N = int(math.isqrt(K))
        ax = axes[0, col]

        ps, qs, energies = [], [], []
        for ch in range(C):
            idx, _ = top_flat_indices(dr, lname, ch, top_k=1)
            flat = idx[0]
            p, q = attribution_to_pq(int(flat), N)
            smap_k = get_smap(dr, lname, ch, int(flat))
            tot = total_smap(dr, lname, ch)
            energies.append(energy_pct(smap_k, tot))
            ps.append(p)
            qs.append(q)

        sc = ax.scatter(qs, ps, c=energies, cmap="YlOrRd",
                        s=max(8, 400 // C), vmin=0, vmax=100, alpha=0.85)
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(-0.5, N - 0.5)
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xlabel("q  (col basis index)", fontsize=8)
        ax.set_ylabel("p  (row basis index)", fontsize=8)
        ax.set_title(f"{lname}  (N={N})", fontsize=9)
        ax.invert_yaxis()
        plt.colorbar(sc, ax=ax, label="Energy %", shrink=0.75)

    fig.suptitle(
        f"Dominant (p,q) per channel — all layers  [{basis_type}]",
        fontsize=11, y=1.02)
    plt.tight_layout()

    out_path = Path(output_dir) / "summary_dominant_pq.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary scatter → {out_path}")


def run_visualisations(
    decomp_results: Dict,
    output_dir:     Path,
    basis_type:     str,
    top_k:          int           = 20,
    image_path:     Optional[str] = None,
    max_channels:   int           = 96,
) -> None:
    dr = decomp_results
    out = Path(output_dir)

    print("\n── Attribution stats CSVs ──")
    write_stats_csvs(dr, out, basis_type, top_k=top_k)

    print("\n── Attribution heatmaps ──")
    write_attribution_heatmaps(dr, out, basis_type)

    print("\n── Dominant component maps ──")
    write_dominant_maps(dr, out, basis_type, max_channels=max_channels)

    print("\n── Top-K component panels ──")
    write_topk_panels(dr, out, basis_type, top_k=top_k, max_channels=max_channels)

    if image_path is not None and "conv1" in dr["layer_components"]:
        print("\n── Conv1 RGB panels ──")
        write_conv1_rgb_panels(dr, out, basis_type, image_path, top_k=top_k)
    else:
        print("\n── Conv1 RGB panels skipped (no image_path) ──")

    print("\n── Summary scatter ──")
    write_summary_scatter(dr, out, basis_type)

    print(f"\nAll visualisations written to {out.resolve()}")


def list_image_paths(
    input_dir: Path,
    extensions: Optional[List[str]] = None,
) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    if extensions is not None:
        exts = {ext.lower() for ext in extensions}

    return sorted(
        path for path in Path(input_dir).iterdir()
        if path.is_file() and path.suffix.lower() in exts
    )


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
    pool_mode:          str            = "softmax",
) -> Dict:
    if alpha_per_pool is None:
        alpha_per_pool = {}

    x = image_tensor.to(DEVICE)
    x_comp = None
    results = {"layer_components": {}, "per_layer_alpha": {}, "layer_errors": []}

    def _record_error(layer_name: str, layer_type: str, components: torch.Tensor, reference: torch.Tensor) -> None:
        error = compute_reconstruction_error(components.sum(-1), reference)
        results["layer_errors"].append({
            "layer": layer_name,
            "type": layer_type,
            "error": error,
        })
        print(f"  {layer_type} {layer_name}: reconstruction error = {error:.6f}")

    relu_idx = 0
    pool_idx = 0
    conv_idx = 0
    conv_names = [n for n, _, _, _ in ALEXNET_CONV_LAYERS]
    pool_names = [n for n, _, _, _ in ALEXNET_POOL_LAYERS]

    with torch.no_grad():
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                pad = layer.padding[0]
                lname = conv_names[conv_idx] if conv_idx < len(conv_names) else f"conv{conv_idx}"

                if basis_type == "none":
                    z_pq = grounding["conv_outputs"][conv_idx].to(DEVICE).unsqueeze(-1)
                else:
                    Nks = layer.kernel_size[0]
                    U = _resolve_1d_basis(basis_type, Nks)
                    if U is None:
                        raise ValueError(f"Could not resolve basis for kernel size {Nks}")
                    if x_comp is None:
                        z_pq = basis_component_maps(
                            x, layer.weight, U,
                            padding=pad, stride=layer.stride[0])
                    else:
                        B, Ci, H, W, K_prev = x_comp.shape
                        z_acc = torch.zeros_like(
                            basis_component_maps(x_comp[..., 0], layer.weight, U,
                                                 padding=pad, stride=layer.stride[0]))
                        for k in range(K_prev):
                            z_acc = z_acc + basis_component_maps(
                                x_comp[..., k], layer.weight, U,
                                padding=pad, stride=layer.stride[0])
                        z_pq = z_acc

                x_comp = z_pq
                if prune is not None:
                    x_comp = prune_components(x_comp, prune)

                results["layer_components"][lname] = x_comp.clone().cpu()
                conv_idx += 1
                x = layer(x)
                _record_error(lname, "conv", x_comp, grounding["conv_outputs"][conv_idx - 1].to(DEVICE))

            elif isinstance(layer, nn.ReLU):
                if x_comp is not None:
                    Z_orig = grounding["conv_outputs"][relu_idx].to(DEVICE)
                    x_comp = relu_decompose(x_comp, Z_orig)
                relu_name = f"relu{relu_idx + 1}"
                relu_idx += 1
                x = layer(x)
                if x_comp is not None:
                    _record_error(relu_name, "relu", x_comp, grounding["relu_outputs"][relu_idx - 1].to(DEVICE))

            elif isinstance(layer, nn.MaxPool2d):
                ks = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
                st = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
                pad_mp = layer.padding if isinstance(layer.padding, int) else layer.padding[0]
                pname = pool_names[pool_idx] if pool_idx < len(pool_names) else f"pool{pool_idx}"
                p_orig = grounding["pool_inputs"][pool_idx].to(DEVICE)

                alpha_val = float(alpha_per_pool[pool_idx]) if alpha_per_pool.get(pool_idx) is not None else estimate_alpha(
                    p_orig, ks, st,
                    target_gap_product=alpha_auto_target,
                    alpha_max=alpha_max,
                    alpha_fallback=alpha_fallback,
                )
                results["per_layer_alpha"][pname] = alpha_val

                if x_comp is not None:
                    if pool_mode == "max_pixel":
                        x_comp = maxpool_decompose_max_pixel(
                            x_comp, p_orig, ks, st, padding=pad_mp)
                    else:  # default softmax mode
                        x_comp = maxpool_decompose(
                            x_comp, p_orig, ks, st,
                            padding=pad_mp, alpha=alpha_val)

                pool_idx += 1
                x = layer(x)
                if x_comp is not None:
                    _record_error(pname, "pool", x_comp, x)

            else:
                x = layer(x)

    results["basis_type"] = basis_type
    results["pool_mode"] = pool_mode
    results["prune"] = prune
    return results


def run_one(
    image_path:        str,
    output_dir:        Path,
    basis_type:        str,
    prune:             Optional[int],
    top_k:             int,
    path_epsilon:      float,
    alpha_auto_target: float,
    alpha_max:         float,
    alpha_fallback:    float,
    alpha_per_pool:    Dict[int, Optional[float]],
    save:              bool,
    visualise:         bool,
    vis_image_path:    Optional[str],
    max_channels:      int,
    pool_mode:         str,
    model,
) -> Dict:
    print(f"\n{'─'*60}")
    print(f"Image : {image_path}")
    print(f"Output: {output_dir}")
    print(f"Basis : {basis_type}  |  prune={prune}  |  top_k={top_k}")
    print(f"{'─'*60}")

    x = preprocess_image(image_path)
    print("Preprocessing image…")

    print("Pass 1: grounding forward pass…")
    grounding = collect_grounding_values(x, model)

    prune_str = f", prune={prune}" if prune is not None else ""
    print(f"Pass 2: decomposed forward (basis={basis_type}{prune_str}, pool={pool_mode})…")
    decomp_results = decomposed_forward(
        x,
        model,
        grounding,
        basis_type=basis_type,
        alpha_per_pool=alpha_per_pool,
        alpha_auto_target=alpha_auto_target,
        alpha_max=alpha_max,
        alpha_fallback=alpha_fallback,
        path_epsilon=path_epsilon,
        prune=prune,
        pool_mode=pool_mode,
    )
    print("  α per pool layer:", decomp_results["per_layer_alpha"])

    print("Computing attributions…")
    target_layers = [name for name, _, _, _ in ALEXNET_CONV_LAYERS]
    attributions = {}
    for layer_name in target_layers:
        if layer_name in decomp_results["layer_components"]:
            attributions[layer_name] = compute_attributions(
                decomp_results["layer_components"],
                target_layer=layer_name,
                path_epsilon=path_epsilon,
                top_k=top_k,
            )
            channel_count = attributions[layer_name]["channel_attributions"].shape[0]
            print(f"  {layer_name}: top-{top_k} components, {channel_count} channels")

    try:
        err = compute_decomposition_error(decomp_results, grounding, layer="conv5", conv_index=4)
        label = ("< 1%" if err < 0.01 else "< 5%" if err < 0.05 else "< 20%" if err < 0.20 else "high")
        print(f"\n  Reconstruction error (normalised L2 @ conv5): {err:.6f}  [{label}]")
        decomp_results["reconstruction_error"] = err
    except (KeyError, IndexError) as exc:
        print(f"  (Error measure skipped: {exc})")

    if save:
        save_outputs(grounding, decomp_results, attributions, output_dir)

    if visualise:
        vip = vis_image_path or image_path
        run_visualisations(
            decomp_results,
            output_dir=output_dir,
            basis_type=basis_type,
            top_k=top_k,
            image_path=vip,
            max_channels=max_channels,
        )

    print("\nDone.")
    print("Layers:", list(decomp_results["layer_components"].keys()))

    return {
        "grounding": grounding,
        "decomp_results": decomp_results,
        "attributions": attributions,
    }


def run_images(
    image_paths: Iterable[Path],
    output_root: Path,
    model,
    basis_type:        str,
    prune:             Optional[int],
    top_k:             int,
    path_epsilon:      float,
    alpha_auto_target: float,
    alpha_max:         float,
    alpha_fallback:    float,
    alpha_per_pool:    Dict[int, Optional[float]],
    save:              bool,
    visualise:         bool,
    vis_image_path:    Optional[str],
    max_channels:      int,
    pool_mode:         str,
    create_subdirs:    bool = False,
) -> None:
    for image_path in image_paths:
        out_dir = output_root / image_path.stem if create_subdirs else output_root
        run_one(
            image_path=str(image_path),
            output_dir=out_dir,
            basis_type=basis_type,
            prune=prune,
            top_k=top_k,
            path_epsilon=path_epsilon,
            alpha_auto_target=alpha_auto_target,
            alpha_max=alpha_max,
            alpha_fallback=alpha_fallback,
            alpha_per_pool=alpha_per_pool,
            save=save,
            visualise=visualise,
            vis_image_path=vis_image_path,
            max_channels=max_channels,
            pool_mode=pool_mode,
            model=model,
        )
