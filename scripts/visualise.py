"""
visualise.py
------------
Inference-time visualisation helpers.  All figures are saved to disk;
matplotlib is set to non-interactive (Agg) backend.

Public API
----------
run_visualisations(decomp_results, grounding, output_dir, cfg)
    Writes all figures under output_dir:
      stats/          — per-layer CSV attribution tables
      heatmaps/       — channel × component heatmaps
      dominant_maps/  — dominant-component spatial map grids
      topk_maps/      — per-channel top-K component panels
      conv1_rgb_panels/ — RGB overlay panels (requires image_path in cfg)
      summary_dominant_pq.png
"""

import csv
import math
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator

from attribution import attribution_to_pq
from basis import haar_basis, dct_basis


# ── Low-level helpers (used internally and by inference scripts) ──────────────

def interp_label(p: int, q: int, basis_type: str) -> str:
    """Human-readable (p, q) label."""
    if basis_type == "none":
        return "(raw)"

    def _ax(v, bt):
        if bt == "haar":
            return "mean" if v == 0 else f"s{v}"
        return "DC" if v == 0 else f"f{v}"

    return f"({_ax(p, basis_type)},{_ax(q, basis_type)})"


def get_smap(dr: Dict, layer: str, channel: int, flat_idx: int) -> np.ndarray:
    """Return (H, W) spatial map for one flat basis index."""
    return dr["layer_components"][layer][0, channel, :, :, flat_idx].numpy()


def energy_pct(smap_k: np.ndarray, smap_total: np.ndarray) -> float:
    """Fraction of total spatial energy explained by one component."""
    total_e = float(np.sum(smap_total ** 2))
    if total_e < 1e-12:
        return 0.0
    return 100.0 * float(np.sum(smap_k ** 2)) / total_e


def top_flat_indices(
    dr:      Dict,
    layer:   str,
    channel: int,
    top_k:   int,
    epsilon: float = 1e-6,
):
    """Return (sorted_flat_indices, mean_abs) for (layer, channel)."""
    comp   = dr["layer_components"][layer]
    mean_a = comp[0, channel].abs().mean(dim=(0, 1)).numpy()   # (K,)
    idx    = np.argsort(mean_a)[::-1][:top_k]
    return idx, mean_a


def total_smap(dr: Dict, layer: str, channel: int) -> np.ndarray:
    """Sum all component maps → should equal grounded activation."""
    comp = dr["layer_components"][layer]
    return comp[0, channel, :, :, :].sum(-1).numpy()


def denorm_image(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation; return uint8 (H, W, 3)."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img  = np.clip(img * std + mean, 0, 1)
    return (img * 255).astype(np.uint8)


def basis_kernel_rgb(U: np.ndarray, p: int, q: int) -> np.ndarray:
    """uint8 (N, N, 3) visualisation of basis element (p, q)."""
    psi  = np.outer(U[p], U[q])
    lo, hi = psi.min(), psi.max()
    psi_n = (psi - lo) / max(hi - lo, 1e-8)
    return (np.stack([psi_n] * 3, axis=-1) * 255).astype(np.uint8)


# ── Per-visualisation functions ───────────────────────────────────────────────

def write_stats_csvs(
    dr:          Dict,
    output_dir:  Path,
    basis_type:  str,
    top_k:       int = 20,
) -> None:
    """Write per-layer attribution CSVs to output_dir/stats/."""
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
                smap_tot    = total_smap(dr, lname, ch)
                total_e     = float(np.sum(smap_tot ** 2)) + 1e-12
                for rank, flat in enumerate(idx, 1):
                    p, q   = attribution_to_pq(int(flat), N)
                    smap_k = get_smap(dr, lname, ch, int(flat))
                    e_pct  = 100.0 * float(np.sum(smap_k ** 2)) / total_e
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
    """Write per-layer channel×component heatmaps to output_dir/heatmaps/."""
    hm_dir = Path(output_dir) / "heatmaps"
    hm_dir.mkdir(parents=True, exist_ok=True)

    for lname, comp in dr["layer_components"].items():
        B, C, H, W, K = comp.shape
        N    = int(math.isqrt(K))
        data = comp[0].abs().mean(dim=(1, 2)).numpy()   # (C, K)

        fig = plt.figure(figsize=(max(14, K // 3), max(5, C // 8 + 2)))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.35)

        ax0 = fig.add_subplot(gs[0])
        im  = ax0.imshow(data, aspect="auto", interpolation="nearest",
                         norm=mcolors.PowerNorm(gamma=0.4), cmap="viridis")
        ax0.set_xlabel(f"Basis component (flat index, N={N})", fontsize=9)
        ax0.set_ylabel("Output channel", fontsize=9)
        ax0.set_title(f"Attribution heatmap — {lname}  [{basis_type}]",
                      fontsize=10)
        ax0.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        plt.colorbar(im, ax=ax0, label="Mean |z|", shrink=0.8)

        ax1      = fig.add_subplot(gs[1])
        marginal = data.mean(axis=0)
        ax1.barh(np.arange(K), marginal[::-1], color="steelblue", height=0.8)
        tick_step = max(1, K // 8)
        ax1.set_yticks(np.arange(0, K, tick_step))
        ax1.set_yticklabels([str(K - 1 - i) for i in range(0, K, tick_step)],
                             fontsize=7)
        ax1.set_xlabel("Mean |z| (avg over channels)", fontsize=8)
        ax1.set_title("Component marginal", fontsize=9)
        ax1.invert_yaxis()

        out_path = hm_dir / f"{lname}_attribution_heatmap.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {lname} heatmap → {out_path.name}")

    print(f"  Heatmaps written to {hm_dir.resolve()}")


def write_dominant_maps(
    dr:              Dict,
    output_dir:      Path,
    basis_type:      str,
    max_channels:    int = 96,
) -> None:
    """Write dominant-component spatial map grids to output_dir/dominant_maps/."""
    dom_dir = Path(output_dir) / "dominant_maps"
    dom_dir.mkdir(parents=True, exist_ok=True)

    for lname, comp in dr["layer_components"].items():
        B, C, H, W, K = comp.shape
        N       = int(math.isqrt(K))
        n_shown = min(C, max_channels)
        n_cols  = min(12, n_shown)
        n_rows  = math.ceil(n_shown / n_cols)
        cell_px = max(1.8, 110 / max(H, W))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * cell_px * 1.05, n_rows * (cell_px + 0.55)))
        axes_flat = np.array(axes).flatten()

        for ch in range(n_shown):
            ax = axes_flat[ch]
            idx, _ = top_flat_indices(dr, lname, ch, top_k=1)
            flat   = idx[0]
            p, q   = attribution_to_pq(int(flat), N)
            smap   = get_smap(dr, lname, ch, int(flat))
            tot    = total_smap(dr, lname, ch)
            e_pct  = energy_pct(smap, tot)
            vmax   = max(np.abs(smap).max(), 1e-8)
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
    """Write per-channel top-K component panels to output_dir/topk_maps/."""
    topk_dir = Path(output_dir) / "topk_maps"

    for lname, comp in dr["layer_components"].items():
        B, C, H, W, K = comp.shape
        N        = int(math.isqrt(K))
        Kd       = min(top_k, K)
        layer_dir = topk_dir / lname
        layer_dir.mkdir(parents=True, exist_ok=True)

        n_ch = min(C, max_channels)
        for ch in range(n_ch):
            idx, mean_a = top_flat_indices(dr, lname, ch, Kd)
            smap_tot    = total_smap(dr, lname, ch)
            total_e     = float(np.sum(smap_tot ** 2)) + 1e-12
            total_shown_e = sum(
                float(np.sum(get_smap(dr, lname, ch, int(f)) ** 2))
                for f in idx) / total_e * 100

            fig      = plt.figure(figsize=(7, max(2, Kd * 0.85 + 0.6)))
            outer_gs = gridspec.GridSpec(Kd, 3, figure=fig,
                                         width_ratios=[0.9, 3, 1.5],
                                         hspace=0.15, wspace=0.08)

            for rank, flat in enumerate(idx):
                p, q   = attribution_to_pq(int(flat), N)
                smap_k = get_smap(dr, lname, ch, int(flat))
                e_pct  = 100.0 * float(np.sum(smap_k ** 2)) / total_e

                ax_lbl = fig.add_subplot(outer_gs[rank, 0])
                ax_lbl.text(0.5, 0.5,
                            f"#{rank+1}\n{interp_label(p, q, basis_type)}\n"
                            f"flat={flat}",
                            ha="center", va="center", fontsize=7,
                            transform=ax_lbl.transAxes)
                ax_lbl.axis("off")

                ax_map = fig.add_subplot(outer_gs[rank, 1])
                vmax   = max(np.abs(smap_k).max(), 1e-8)
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
    """
    Write conv1 RGB overlay panels to output_dir/conv1_rgb_panels/.
    Requires the original image path so it can overlay component maps.
    """
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset
    except ImportError:
        _inset = None

    from attribution import preprocess_image

    rgb_dir = Path(output_dir) / "conv1_rgb_panels"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    x_tensor = preprocess_image(image_path)
    img_rgb  = denorm_image(x_tensor)

    comp1 = dr["layer_components"]["conv1"]
    B, C, Ho, Wo, K = comp1.shape
    N1 = int(math.isqrt(K))
    U1 = haar_basis(N1) if basis_type == "haar" else dct_basis(N1)
    Kd = min(top_k, K)
    H_in, W_in = 224, 224

    for ch in range(C):
        idx, mean_a = top_flat_indices(dr, "conv1", ch, Kd)
        smap_tot    = total_smap(dr, "conv1", ch)
        total_e     = float(np.sum(smap_tot ** 2)) + 1e-12

        n_rows  = Kd + 1
        fig     = plt.figure(figsize=(11, n_rows * 1.45 + 0.8))
        gs_main = gridspec.GridSpec(n_rows, 4, figure=fig,
                                    width_ratios=[1.1, 2.8, 2.0, 1.5],
                                    hspace=0.08, wspace=0.06)

        # Row 0: original image + total output map
        ax_lbl = fig.add_subplot(gs_main[0, 0])
        ax_lbl.text(0.5, 0.5, f"ch {ch}\nOriginal",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    transform=ax_lbl.transAxes)
        ax_lbl.axis("off")

        ax_img = fig.add_subplot(gs_main[0, 1])
        ax_img.imshow(img_rgb)
        ax_img.set_title("Original image (224×224)", fontsize=8)
        ax_img.axis("off")

        ax_tot  = fig.add_subplot(gs_main[0, 2])
        vmax_t  = max(np.abs(smap_tot).max(), 1e-8)
        ax_tot.imshow(smap_tot, cmap="RdBu_r", vmin=-vmax_t, vmax=vmax_t,
                      interpolation="nearest")
        ax_tot.set_title(f"Conv1 ch{ch} total output (55×55)", fontsize=8)
        ax_tot.axis("off")
        fig.add_subplot(gs_main[0, 3]).axis("off")

        # Rows 1..Kd: top-K components
        top1_e = (100. * float(np.sum(
            get_smap(dr, "conv1", ch, int(idx[0])) ** 2)) / total_e)

        for rank, flat in enumerate(idx):
            row    = rank + 1
            p, q   = attribution_to_pq(int(flat), N1)
            smap_k = get_smap(dr, "conv1", ch, int(flat))
            e_pct  = 100.0 * float(np.sum(smap_k ** 2)) / total_e

            psi_vis = basis_kernel_rgb(U1, p, q)

            smap_t  = torch.tensor(smap_k).unsqueeze(0).unsqueeze(0).float()
            smap_up = F.interpolate(smap_t, size=(H_in, W_in),
                                    mode="bilinear", align_corners=False)
            smap_up = smap_up.squeeze().numpy()

            vmax_up = max(np.abs(smap_up).max(), 1e-8)
            alpha   = np.clip((np.abs(smap_up) / vmax_up) * 0.65, 0, 0.65)
            sgn     = np.sign(smap_up)
            r_ch    = np.clip(img_rgb[:, :, 0] / 255. + alpha * np.clip(sgn,  0, 1) * 0.6, 0, 1)
            b_ch    = np.clip(img_rgb[:, :, 2] / 255. + alpha * np.clip(-sgn, 0, 1) * 0.6, 0, 1)
            g_ch    = np.clip(img_rgb[:, :, 1] / 255. * (1 - alpha * 0.4),  0, 1)
            composite = np.stack([r_ch, g_ch, b_ch], axis=-1)

            ax_lbl = fig.add_subplot(gs_main[row, 0])
            ax_lbl.text(0.5, 0.5,
                        f"#{rank+1}\n{interp_label(p, q, basis_type)}\n"
                        f"flat={flat}",
                        ha="center", va="center", fontsize=7,
                        transform=ax_lbl.transAxes)
            ax_lbl.axis("off")

            ax_comp = fig.add_subplot(gs_main[row, 1])
            ax_comp.imshow(composite, interpolation="bilinear")
            if _inset is not None:
                axins = _inset(ax_comp, width="22%", height="22%",
                               loc="upper right")
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
            100. * float(np.sum(get_smap(dr, "conv1", ch, int(f)) ** 2)) / total_e
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
    """Write dominant-(p,q)-per-channel scatter plot for all layers."""
    n_layers = len(dr["layer_components"])
    fig, axes = plt.subplots(
        1, n_layers,
        figsize=(max(5 * n_layers, 8), 5),
        squeeze=False)

    for col, (lname, comp) in enumerate(dr["layer_components"].items()):
        B, C, H, W, K = comp.shape
        N  = int(math.isqrt(K))
        ax = axes[0, col]

        ps, qs, energies = [], [], []
        for ch in range(C):
            idx, _ = top_flat_indices(dr, lname, ch, top_k=1)
            flat   = idx[0]
            p, q   = attribution_to_pq(int(flat), N)
            smap_k = get_smap(dr, lname, ch, int(flat))
            tot    = total_smap(dr, lname, ch)
            e      = energy_pct(smap_k, tot)
            ps.append(p); qs.append(q); energies.append(e)

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


# ── Top-level entry point ─────────────────────────────────────────────────────

def run_visualisations(
    decomp_results: Dict,
    output_dir:     Path,
    basis_type:     str,
    top_k:          int           = 20,
    image_path:     Optional[str] = None,
    max_channels:   int           = 96,
) -> None:
    """
    Run all visualisations and write results under output_dir.

    Parameters
    ----------
    decomp_results : dict from decomposed_forward / load_outputs
    output_dir     : root directory for all figure output
    basis_type     : "haar" | "dct" | "none"
    top_k          : number of top components per channel / panel
    image_path     : path to original image (required for conv1 RGB panels;
                     skip those panels if None)
    max_channels   : cap channels written per layer (dominant + topk panels)
    """
    dr  = decomp_results
    out = Path(output_dir)

    print("\n── Attribution stats CSVs ──")
    write_stats_csvs(dr, out, basis_type, top_k=top_k)

    print("\n── Attribution heatmaps ──")
    write_attribution_heatmaps(dr, out, basis_type)

    print("\n── Dominant component maps ──")
    write_dominant_maps(dr, out, basis_type, max_channels=max_channels)

    print("\n── Top-K component panels ──")
    write_topk_panels(dr, out, basis_type,
                      top_k=top_k, max_channels=max_channels)

    if image_path is not None and "conv1" in dr["layer_components"]:
        print("\n── Conv1 RGB panels ──")
        write_conv1_rgb_panels(dr, out, basis_type, image_path, top_k=top_k)
    else:
        print("\n── Conv1 RGB panels skipped (no image_path) ──")

    print("\n── Summary scatter ──")
    write_summary_scatter(dr, out, basis_type)

    print(f"\nAll visualisations written to {out.resolve()}")
