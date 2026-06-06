import argparse
import sys
from pathlib import Path

# Ensure the scripts directory is importable when run from the repository root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

from pipeline import collect_grounding_values, decomposed_forward, preprocess_image


def disable_conv_biases(model: nn.Module) -> None:
    for layer in model.features:
        if isinstance(layer, nn.Conv2d) and layer.bias is not None:
            layer.bias = None
            layer.register_parameter("bias", None)


def load_alexnet_bias_false(pretrained: bool = True) -> nn.Module:
    try:
        weights = tv_models.AlexNet_Weights.DEFAULT if pretrained else None
        model = tv_models.alexnet(weights=weights)
    except AttributeError:
        model = tv_models.alexnet(pretrained=pretrained)
    disable_conv_biases(model)
    model.eval()
    return model


def find_example_image(root: Path) -> Path:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in extensions:
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test reconstruction error with grounding run using bias=False for AlexNet conv layers."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to an input image. If omitted, a random tensor is used.",
    )
    parser.add_argument(
        "--basis",
        default="haar",
        choices=["haar", "dct", "none"],
        help="Basis type used for kernel decomposition.",
    )
    parser.add_argument(
        "--pool-mode",
        default="softmax",
        choices=["softmax", "max_pixel"],
        help="MaxPool decomposition mode.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not use pretrained AlexNet weights.",
    )
    args = parser.parse_args()

    if args.image_path:
        image_path = Path(args.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        x = preprocess_image(str(image_path))
    else:
        example = find_example_image(Path(__file__).resolve().parents[1])
        if example is not None:
            print(f"Using example image: {example}")
            x = preprocess_image(str(example))
        else:
            print("No image found; using random input tensor.")
            x = torch.randn(1, 3, 224, 224)

    model = load_alexnet_bias_false(pretrained=not args.no_pretrained)
    grounding = collect_grounding_values(x, model)
    results = decomposed_forward(
        x,
        model,
        grounding,
        basis_type=args.basis,
        pool_mode=args.pool_mode,
    )

    print("\nReconstruction error per layer:")
    for entry in results["layer_errors"]:
        print(f"  {entry['type']:>4} {entry['layer']:7}: {entry['error']:.6f}")

    if "reconstruction_error" in results:
        print(f"\nFinal stored reconstruction error: {results['reconstruction_error']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
