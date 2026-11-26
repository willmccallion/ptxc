"""Input sets: directories of `input_k.pb` tensors, one per graph input in
order, identified by a hash of their contents so a changed set is a cache
miss (TESTING.md)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

#: The file each set's description is written to.
DESCRIPTION = "input_set.json"


def input_file(directory: Path, index: int) -> Path:
    return directory / f"input_{index}.pb"


def write_set(directory: Path, tensors: list[np.ndarray], description: dict) -> str:
    """Writes `tensors` as `input_k.pb` files and returns the set's id."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, tensor in enumerate(tensors):
        proto = numpy_helper.from_array(np.ascontiguousarray(tensor))
        input_file(directory, index).write_bytes(proto.SerializeToString())
    set_id = input_set_id(directory)
    (directory / DESCRIPTION).write_text(json.dumps({"id": set_id, **description}, indent=2) + "\n")
    return set_id


def read_set(directory: Path) -> list[np.ndarray]:
    """The tensors of the set in `directory`, in input order."""
    tensors = []
    index = 0
    while (path := input_file(directory, index)).exists():
        tensors.append(numpy_helper.to_array(onnx.load_tensor(str(path))))
        index += 1
    return tensors


def input_set_id(directory: Path) -> str:
    """SHA-256, in hex, over each `input_k.pb` in order: its length as 8
    little-endian bytes, then its bytes. `ptxc-testkit` computes the same."""
    digest = hashlib.sha256()
    index = 0
    while (path := input_file(directory, index)).exists():
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
        index += 1
    if index == 0:
        raise FileNotFoundError(f"{directory} holds no input_0.pb")
    return digest.hexdigest()


#: Tokenizer vocabulary sizes, so random token ids are valid.
BERT_VOCABULARY = 30_522
GPT2_VOCABULARY = 50_257


def random_inputs(model: str, batch: int, sequence: int, seed: int) -> list[np.ndarray]:
    """Seeded, valid inputs for one of the target models: normal floats for
    the MLP and ResNet-50, token ids within the vocabulary for BERT and
    GPT-2 (with BERT's mask all ones and its token types all zeros)."""
    rng = np.random.default_rng(seed)
    if model == "mlp":
        return [rng.standard_normal((batch, 784), dtype=np.float32)]
    if model in ("resnet50", "resnet50-unfolded"):
        return [rng.standard_normal((batch, 3, 224, 224), dtype=np.float32)]
    shape = (batch, sequence)
    if model == "bert-base":
        ids = rng.integers(0, BERT_VOCABULARY, shape, dtype=np.int64)
        return [ids, np.ones(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)]
    if model == "gpt2":
        return [rng.integers(0, GPT2_VOCABULARY, shape, dtype=np.int64)]
    raise ValueError(f"no input recipe for {model!r}")


def imagenet_v2_names(directory: Path, count: int) -> list[str]:
    """`count` evenly spaced ImageNetV2 images, as `<class>/<name>` paths,
    from its files sorted by class number and then name."""
    files = sorted(directory.glob("*/*"), key=lambda path: (int(path.parent.name), path.name))
    if len(files) < count:
        raise ValueError(f"{directory} holds {len(files)} images, fewer than {count}")
    return [
        f"{files[i * len(files) // count].parent.name}/{files[i * len(files) // count].name}"
        for i in range(count)
    ]


def imagenet_inputs(directory: Path, names: list[str]) -> np.ndarray:
    """The images, preprocessed as ResNet-50's IMAGENET1K_V1 weights expect,
    stacked into one `[N, 3, 224, 224]` batch."""
    from PIL import Image
    from torchvision.models import ResNet50_Weights

    transform = ResNet50_Weights.IMAGENET1K_V1.transforms()
    images = [transform(Image.open(directory / name).convert("RGB")).numpy() for name in names]
    return np.stack(images)
