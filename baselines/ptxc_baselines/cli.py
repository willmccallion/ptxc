"""`ptxc-baselines`: exports the target models, makes input sets, and caches
ONNX Runtime's reference outputs."""

import argparse
import sys
from pathlib import Path

from . import inputs, models, paths, reference

#: Each image source: its directory, the set name's prefix, and how to get it.
IMAGE_SOURCES = {
    "imagenetv2": (paths.IMAGENET_V2, "imagenetv2", "run `cargo xtask fetch imagenetv2`"),
    "imagenet": (
        paths.IMAGENET_VAL,
        "imagenet-val",
        "download ILSVRC2012_img_val.tar and extract it there (D25)",
    ),
}


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ptxc-baselines", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="export a target model to data/models/<name>/")
    export.add_argument("--model", required=True, choices=sorted(models.MODELS))
    images = commands.add_parser(
        "images", help="make an input set of preprocessed images for ResNet-50"
    )
    images.add_argument("--source", required=True, choices=sorted(IMAGE_SOURCES))
    images.add_argument("--count", type=int, default=16)
    random = commands.add_parser("random", help="make a seeded random input set for a target model")
    random.add_argument("--model", required=True, choices=sorted(models.MODELS))
    random.add_argument("--batch", type=int, default=1)
    random.add_argument("--sequence", type=int, default=128)
    random.add_argument("--seed", type=int, default=0)
    ort = commands.add_parser(
        "reference", help="cache ONNX Runtime's outputs for a model and input set"
    )
    ort.add_argument("--model", required=True, help="a target model's name, or an ONNX file")
    ort.add_argument(
        "--inputs", required=True, help="an input set's name in data/inputs/, or a directory"
    )
    return parser.parse_args(argv)


def run_export(args: argparse.Namespace) -> Path:
    path = paths.model_path(args.model)
    models.export(models.MODELS[args.model](), path)
    return path


def select_images(source: str, directory: Path, count: int) -> list[str]:
    """ImageNetV2's evenly spaced images (D35), or the first of D25's list."""
    if source == "imagenetv2":
        return inputs.imagenet_v2_names(directory, count)
    return paths.IMAGENET_LIST.read_text().split()[:count]


def run_images(args: argparse.Namespace) -> Path:
    directory, prefix, fetch = IMAGE_SOURCES[args.source]
    if not directory.is_dir():
        sys.exit(f"{directory} does not exist: {fetch}")
    names = select_images(args.source, directory, args.count)
    out = paths.INPUTS / f"{prefix}-{args.count}"
    description = {
        "kind": args.source,
        "images": names,
        "preprocessing": "ResNet50_Weights.IMAGENET1K_V1",
    }
    inputs.write_set(out, [inputs.imagenet_inputs(directory, names)], description)
    return out


def run_random(args: argparse.Namespace) -> Path:
    out = paths.INPUTS / f"{args.model}-batch{args.batch}-sequence{args.sequence}-seed{args.seed}"
    tensors = inputs.random_inputs(args.model, args.batch, args.sequence, args.seed)
    description = {
        "kind": "random",
        "model": args.model,
        "batch": args.batch,
        "sequence": args.sequence,
        "seed": args.seed,
    }
    inputs.write_set(out, tensors, description)
    return out


def run_reference(args: argparse.Namespace) -> Path:
    model = paths.model_path(args.model) if args.model in models.MODELS else Path(args.model)
    inputs_dir = Path(args.inputs) if Path(args.inputs).is_dir() else paths.INPUTS / args.inputs
    return reference.cache(model, inputs_dir)


COMMANDS = {
    "export": run_export,
    "images": run_images,
    "random": run_random,
    "reference": run_reference,
}


def main(argv: list[str] | None = None) -> None:
    args = parse(sys.argv[1:] if argv is None else argv)
    print(COMMANDS[args.command](args))
