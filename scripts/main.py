"""
main.py
-------
CLI entry point for the CNN basis-decomposition pipeline.
"""

import argparse
import sys
from pathlib import Path

from pipeline import load_alexnet, list_image_paths, run_images


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="main.py",
        description="CNN multiresolution basis-path decomposition (AlexNet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("I/O")
    io.add_argument(
        "image",
        nargs="?",
        metavar="IMAGE",
        help="Path to a single input image. Mutually exclusive with --input-dir.",
    )
    io.add_argument(
        "--input-dir", "-i",
        metavar="DIR",
        help="Process all images in DIR (jpg/png/jpeg/bmp/tiff).",
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

    basis = p.add_argument_group("Basis")
    basis.add_argument(
        "--basis", "-b",
        choices=["haar", "dct", "none"],
        default="haar",
        help='Kernel decomposition basis. "none" skips decomposition (K=1).',
    )

    prune = p.add_argument_group("Pruning")
    prune.add_argument(
        "--prune",
        type=int,
        default=None,
        metavar="N",
        help="Keep only the top-N components per channel after each conv.",
    )

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

    pool = p.add_argument_group("MaxPool softmax α")
    pool.add_argument(
        "--alpha-auto-target",
        type=float,
        default=30.0,
        metavar="T",
        help="Target α × gap for automatic α selection.",
    )
    pool.add_argument(
        "--alpha-max",
        type=float,
        default=200.0,
        metavar="A",
        help="Hard ceiling on α.",
    )
    pool.add_argument(
        "--alpha-fallback",
        type=float,
        default=50.0,
        metavar="A",
        help="α used when gap estimation fails.",
    )
    pool.add_argument(
        "--alpha-pool0",
        type=float,
        default=None,
        metavar="A",
        help="Manual α for pool1. None = auto.",
    )
    pool.add_argument(
        "--alpha-pool1",
        type=float,
        default=None,
        metavar="A",
        help="Manual α for pool2. None = auto.",
    )
    pool.add_argument(
        "--pool-mode",
        choices=["softmax", "max_pixel"],
        default="max-pixel",
        help='MaxPool decomposition mode: "softmax" (default, softmax approximation) or "max_pixel" (direct max pixel pass-through).',
    )

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
        help="Path to the original image for conv1 RGB overlay panels.",
    )
    vis.add_argument(
        "--max-channels",
        type=int,
        default=96,
        metavar="N",
        help="Maximum channels shown per layer in visualisations.",
    )

    model_g = p.add_argument_group("Model")
    model_g.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use randomly initialised AlexNet.",
    )

    return p.parse_args(argv)


def _build_alpha_overrides(args) -> dict:
    return {
        0: args.alpha_pool0,
        1: args.alpha_pool1,
    }


def main(argv=None):
    args = parse_args(argv)

    if args.image and args.input_dir:
        print("Error: provide either a positional IMAGE or --input-dir, not both.", file=sys.stderr)
        sys.exit(1)
    if not args.image and not args.input_dir:
        print("Error: provide either a positional IMAGE or --input-dir.", file=sys.stderr)
        sys.exit(1)

    if args.input_dir:
        images = list_image_paths(Path(args.input_dir))
        if not images:
            print(f"Error: no images found in {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(images)} image(s) in {args.input_dir}")
    else:
        images = [Path(args.image)]

    alpha_per_pool = _build_alpha_overrides(args)
    model = load_alexnet(pretrained=not args.no_pretrained)
    output_root = Path(args.output_dir)

    run_images(
        image_paths=images,
        output_root=output_root,
        model=model,
        basis_type=args.basis,
        prune=args.prune,
        top_k=args.top_k,
        path_epsilon=args.path_epsilon,
        alpha_auto_target=args.alpha_auto_target,
        alpha_max=args.alpha_max,
        alpha_fallback=args.alpha_fallback,
        alpha_per_pool=alpha_per_pool,
        save=not args.no_save,
        visualise=not args.no_visualise,
        vis_image_path=args.image_path,
        max_channels=args.max_channels,
        pool_mode=args.pool_mode,
        create_subdirs=bool(args.input_dir),
    )


if __name__ == "__main__":
    main()
