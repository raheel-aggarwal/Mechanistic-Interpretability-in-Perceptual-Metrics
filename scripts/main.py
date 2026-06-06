"""
main.py
-------
CLI entry point for the CNN basis-decomposition pipeline.

Usage examples
--------------
# Run on a single image with defaults:
python main.py path/to/image.jpg

# Full options:
python main.py path/to/image.jpg \\
    --output-dir ./results \\
    --basis haar \\
    --prune 5 \\
    --top-k 30 \\
    --alpha-auto-target 30.0 \\
    --alpha-max 200.0 \\
    --alpha-fallback 50.0 \\
    --path-epsilon 1e-6 \\
    --no-save \\
    --no-visualise

# Use DCT basis, no pruning, run visualisations with original image:
python main.py img.jpg --basis dct --image-path img.jpg --output-dir out/

# No kernel decomposition (basis=none, K=1 passthrough):
python main.py img.jpg --basis none --output-dir out_none/

# Process all images in a folder:
python main.py --input-dir ./images --output-dir ./results --basis haar
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="main.py",
        description="CNN multiresolution basis-path decomposition (AlexNet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    io = p.add_argument_group("I/O")
    io.add_argument(
        "image",
        nargs="?",
        metavar="IMAGE",
        help="Path to a single input image.  "
             "Mutually exclusive with --input-dir.",
    )
    io.add_argument(
        "--input-dir", "-i",
        metavar="DIR",
        help="Process all images in DIR (jpg/png/jpeg/bmp/tiff).  "
             "Mutually exclusive with positional IMAGE.",
    )
    io.add_argument(
        "--output-dir", "-o",
        default="./decomp_outputs",
        metavar="DIR",
        help="Root directory for all saved outputs.",
    )
    io.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving grounding.pkl / components.pkl / attributions.json.",
    )

    # ── Basis ─────────────────────────────────────────────────────────────────
    basis = p.add_argument_group("Basis")
    basis.add_argument(
        "--basis", "-b",
        choices=["haar", "dct", "none"],
        default="haar",
        help='Kernel decomposition basis. "none" skips decomposition (K=1).',
    )

    # ── Pruning ───────────────────────────────────────────────────────────────
    prune = p.add_argument_group("Pruning")
    prune.add_argument(
        "--prune",
        type=int,
        default=None,
        metavar="N",
        help="Keep only the top-N components per channel after each conv.  "
             "None = no pruning.",
    )

    # ── Attribution ───────────────────────────────────────────────────────────
    attr = p.add_argument_group("Attribution")
    attr.add_argument(
        "--top-k",
        type=int,
        default=20,
        metavar="K",
        help="Number of top components recorded per channel in attributions.",
    )
    attr.add_argument(
        "--path-epsilon",
        type=float,
        default=1e-6,
        metavar="EPS",
        help="Drop attribution components with mean |z| below this threshold.",
    )

    # ── MaxPool α ─────────────────────────────────────────────────────────────
    pool = p.add_argument_group("MaxPool softmax α")
    pool.add_argument(
        "--alpha-auto-target",
        type=float,
        default=30.0,
        metavar="T",
        help="Target α × gap for automatic α selection (higher = more accurate).",
    )
    pool.add_argument(
        "--alpha-max",
        type=float,
        default=200.0,
        metavar="A",
        help="Hard ceiling on α (prevents float32 overflow).",
    )
    pool.add_argument(
        "--alpha-fallback",
        type=float,
        default=50.0,
        metavar="A",
        help="α used when gap estimation fails (near-tied pooling inputs).",
    )
    pool.add_argument(
        "--alpha-pool0",
        type=float,
        default=None,
        metavar="A",
        help="Manual α for pool1 (overrides auto).  None = auto.",
    )
    pool.add_argument(
        "--alpha-pool1",
        type=float,
        default=None,
        metavar="A",
        help="Manual α for pool2 (overrides auto).  None = auto.",
    )

    # ── Visualisation ─────────────────────────────────────────────────────────
    vis = p.add_argument_group("Visualisation")
    vis.add_argument(
        "--no-visualise",
        action="store_true",
        help="Skip all figure generation.",
    )
    vis.add_argument(
        "--image-path",
        metavar="PATH",
        default=None,
        help="Path to the original image for conv1 RGB overlay panels.  "
             "Defaults to the input image when processing a single file.",
    )
    vis.add_argument(
        "--max-channels",
        type=int,
        default=96,
        metavar="N",
        help="Maximum channels shown per layer in dominant / top-K panels.",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_g = p.add_argument_group("Model")
    model_g.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use randomly initialised AlexNet (for debugging).",
    )

    return p.parse_args(argv)


# ── Pipeline runner ───────────────────────────────────────────────────────────

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
    alpha_per_pool:    dict,
    save:              bool,
    visualise:         bool,
    vis_image_path:    Optional[str],
    max_channels:      int,
    model,
) -> None:
    """Run the full pipeline for one image."""
    # Deferred imports so each module is independently importable
    from attribution        import preprocess_image, compute_attributions, compute_decomposition_error
    from model              import collect_grounding_values, ALEXNET_CONV_LAYERS
    from decomposed_forward import decomposed_forward
    from io_utils           import save_outputs
    from visualise          import run_visualisations

    print(f"\n{'─'*60}")
    print(f"Image : {image_path}")
    print(f"Output: {output_dir}")
    print(f"Basis : {basis_type}  |  prune={prune}  |  top_k={top_k}")
    print(f"{'─'*60}")

    # Preprocessing
    print("Preprocessing image…")
    x = preprocess_image(image_path)

    # Pass 1 — grounding
    print("Pass 1: grounding forward pass…")
    grounding = collect_grounding_values(x, model)

    # Pass 2 — decomposed forward
    prune_str = f", prune={prune}" if prune is not None else ""
    print(f"Pass 2: decomposed forward (basis={basis_type}{prune_str})…")
    decomp_results = decomposed_forward(
        x, model, grounding,
        basis_type        = basis_type,
        alpha_per_pool    = alpha_per_pool,
        alpha_auto_target = alpha_auto_target,
        alpha_max         = alpha_max,
        alpha_fallback    = alpha_fallback,
        path_epsilon      = path_epsilon,
        prune             = prune,
    )
    print("  α per pool layer:", decomp_results["per_layer_alpha"])

    # Attributions
    print("Computing attributions…")
    target_layers = [n for n, _, _, _ in ALEXNET_CONV_LAYERS]
    attributions  = {}
    for lname in target_layers:
        if lname in decomp_results["layer_components"]:
            attributions[lname] = compute_attributions(
                decomp_results["layer_components"],
                target_layer = lname,
                path_epsilon = path_epsilon,
                top_k        = top_k,
            )
            C = attributions[lname]["channel_attributions"].shape[0]
            print(f"  {lname}: top-{top_k} components, {C} channels")

    # Reconstruction error
    try:
        err = compute_decomposition_error(
            decomp_results, grounding, layer="conv5", conv_index=4)
        label = ("< 1%" if err < 0.01 else
                 "< 5%" if err < 0.05 else
                 "< 20%" if err < 0.20 else "high")
        print(f"\n  Reconstruction error (normalised L2 @ conv5): "
              f"{err:.6f}  [{label}]")
        decomp_results["reconstruction_error"] = err
    except (KeyError, IndexError) as e:
        print(f"  (Error measure skipped: {e})")

    # Save
    if save:
        save_outputs(grounding, decomp_results, attributions, output_dir)

    # Visualise
    if visualise:
        vip = vis_image_path or image_path
        run_visualisations(
            decomp_results,
            output_dir  = output_dir,
            basis_type  = basis_type,
            top_k       = top_k,
            image_path  = vip,
            max_channels = max_channels,
        )

    print("\nDone.")
    print("Layers:", list(decomp_results["layer_components"].keys()))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)

    # Validate I/O arguments
    if args.image and args.input_dir:
        print("Error: provide either a positional IMAGE or --input-dir, not both.",
              file=sys.stderr)
        sys.exit(1)
    if not args.image and not args.input_dir:
        print("Error: provide either a positional IMAGE or --input-dir.",
              file=sys.stderr)
        sys.exit(1)

    # Collect image paths
    if args.input_dir:
        exts   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        images = sorted(
            p for p in Path(args.input_dir).iterdir()
            if p.suffix.lower() in exts)
        if not images:
            print(f"Error: no images found in {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(images)} image(s) in {args.input_dir}")
    else:
        images = [Path(args.image)]

    # alpha overrides
    alpha_per_pool = {
        0: args.alpha_pool0,
        1: args.alpha_pool1,
    }

    # Load model once and reuse across all images
    from model import load_alexnet
    model = load_alexnet(pretrained=not args.no_pretrained)

    output_root = Path(args.output_dir)

    for img_path in images:
        # If processing a folder, nest each image in its own sub-directory
        if args.input_dir:
            out_dir = output_root / img_path.stem
        else:
            out_dir = output_root

        run_one(
            image_path        = str(img_path),
            output_dir        = out_dir,
            basis_type        = args.basis,
            prune             = args.prune,
            top_k             = args.top_k,
            path_epsilon      = args.path_epsilon,
            alpha_auto_target = args.alpha_auto_target,
            alpha_max         = args.alpha_max,
            alpha_fallback    = args.alpha_fallback,
            alpha_per_pool    = alpha_per_pool,
            save              = not args.no_save,
            visualise         = not args.no_visualise,
            vis_image_path    = args.image_path,
            max_channels      = args.max_channels,
            model             = model,
        )


if __name__ == "__main__":
    main()
