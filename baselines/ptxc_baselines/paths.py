"""Where ptxc keeps models, input sets and reference outputs (TESTING.md)."""

from pathlib import Path

#: The repository root: this package lives in `baselines/ptxc_baselines/`.
ROOT = Path(__file__).resolve().parents[2]

#: Downloaded and generated data, gitignored.
DATA = ROOT / "data"

#: Exported ONNX models, one directory per model.
MODELS = DATA / "models"

#: Input sets, one directory of `input_k.pb` files each.
INPUTS = DATA / "inputs"

#: Cached reference outputs, keyed by model, input set and tool.
REFERENCE = DATA / "reference"

#: The ImageNet validation images, downloaded by hand (D25).
IMAGENET_VAL = DATA / "datasets" / "imagenet-val"

#: ImageNetV2's matched-frequency images, an interim stand-in (D35), which
#: `cargo xtask fetch imagenetv2` downloads: `<class>/<name>.jpeg`.
IMAGENET_V2 = DATA / "datasets" / "imagenetv2"

#: The 5,000 validation file names used (D25): every tenth image.
IMAGENET_LIST = Path(__file__).resolve().parent / "imagenet_val_5000.txt"


def model_path(name: str) -> Path:
    """The ONNX file of the exported model `name`."""
    return MODELS / name / "model.onnx"
