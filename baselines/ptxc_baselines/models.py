"""The target models (MLP, ResNet-50, BERT-base, GPT-2), exported to ONNX
with named `batch` and `sequence` dims that ptxc binds at import (D3)."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.export import Dim

#: The opset every export uses, the newest the ONNX pin (D31) defines fully.
OPSET = 21

#: Hugging Face repositories and the revisions their weights are pinned to.
BERT = ("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594")
GPT2 = ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e")

#: The example sizes the exporter traces with; the dims stay symbolic. Both
#: are above 1, which torch.export would specialise the dim to.
EXAMPLE_BATCH = 2
EXAMPLE_SEQUENCE = 16


@dataclass(frozen=True)
class Export:
    """What to export: a module, example inputs and the names of things."""

    module: torch.nn.Module
    inputs: tuple[torch.Tensor, ...]
    input_names: list[str]
    output_names: list[str]
    #: For each input, the axes that are symbolic, by name.
    dims: list[dict[int, str]]
    #: Whether the exporter's optimiser runs, which among other things
    #: folds each batch normalisation into the convolution before it.
    optimize: bool = True


class Mlp(torch.nn.Module):
    """A three-layer MLP on flattened 28x28 images, with seeded weights."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HiddenStates(torch.nn.Module):
    """BERT's final hidden states and pooled output, as a tuple."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        return out.last_hidden_state, out.pooler_output


class Logits(torch.nn.Module):
    """GPT-2's next-token logits, without a key-value cache."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, use_cache=False).logits


def mlp() -> Export:
    return Export(
        Mlp().eval(), (torch.randn(EXAMPLE_BATCH, 784),), ["x"], ["logits"], [{0: "batch"}]
    )


def resnet50() -> Export:
    from torchvision.models import ResNet50_Weights
    from torchvision.models import resnet50 as build

    module = build(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
    image = torch.randn(EXAMPLE_BATCH, 3, 224, 224)
    return Export(module, (image,), ["image"], ["logits"], [{0: "batch"}])


def resnet50_unfolded() -> Export:
    """ResNet-50 with its batch normalisations kept, for ptxc to fold."""
    return dataclasses.replace(resnet50(), optimize=False)


def tokens() -> torch.Tensor:
    return torch.randint(0, 1000, (EXAMPLE_BATCH, EXAMPLE_SEQUENCE))


def bert_base() -> Export:
    from transformers import BertModel

    name, revision = BERT
    model = BertModel.from_pretrained(name, revision=revision, attn_implementation="eager").eval()
    ids = tokens()
    inputs = (ids, torch.ones_like(ids), torch.zeros_like(ids))
    sequence = {0: "batch", 1: "sequence"}
    names = ["input_ids", "attention_mask", "token_type_ids"]
    return Export(
        HiddenStates(model).eval(),
        inputs,
        names,
        ["last_hidden_state", "pooler_output"],
        [sequence] * 3,
    )


def gpt2() -> Export:
    from transformers import GPT2LMHeadModel

    name, revision = GPT2
    model = GPT2LMHeadModel.from_pretrained(
        name, revision=revision, attn_implementation="eager"
    ).eval()
    return Export(
        Logits(model).eval(), (tokens(),), ["input_ids"], ["logits"], [{0: "batch", 1: "sequence"}]
    )


#: Every target model, by the name `export` takes.
MODELS: dict[str, Callable[[], Export]] = {
    "mlp": mlp,
    "resnet50": resnet50,
    "resnet50-unfolded": resnet50_unfolded,
    "bert-base": bert_base,
    "gpt2": gpt2,
}


def export(spec: Export, path: Path) -> None:
    """Exports `spec` to `path` as one self-contained ONNX file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    named = {name: Dim(name) for axes in spec.dims for name in axes.values()}
    # A tuple in input order: a dict would have to use forward's argument names.
    dynamic = tuple({axis: named[dim] for axis, dim in axes.items()} for axes in spec.dims)
    with torch.no_grad():
        program = torch.onnx.export(
            spec.module,
            spec.inputs,
            dynamo=True,
            opset_version=OPSET,
            input_names=spec.input_names,
            output_names=spec.output_names,
            dynamic_shapes=dynamic,
            optimize=spec.optimize,
        )
    program.save(str(path), external_data=False)
