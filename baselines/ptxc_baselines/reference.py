"""Reference outputs from ONNX Runtime, cached by model, input set and
tool version (TESTING.md), as `output_k.pb` tensors."""

import hashlib
import json
import platform
import sys
from pathlib import Path

import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented as NoKernel

from .inputs import input_set_id, read_set
from .paths import REFERENCE

#: The file each reference's provenance is written to.
MANIFEST = "manifest.json"

#: Each exported fuzz case's model and input set, in its directory, as
#: `ptxc fuzz-export` writes them (D54).
CASE_MODEL = "model.onnx"
CASE_INPUTS = "inputs"

#: No graph rewrites, so ONNX Runtime computes the model as written.
OPTIMIZATION = ort.GraphOptimizationLevel.ORT_DISABLE_ALL


def model_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_dir(model: Path, inputs: Path) -> Path:
    """Where ONNX Runtime's outputs for `model` on `inputs` are cached."""
    return REFERENCE / model_sha256(model) / input_set_id(inputs) / f"ort-{ort.__version__}"


def open_session(model: Path) -> ort.InferenceSession:
    """An ONNX Runtime session for `model` on the CPU, without graph rewrites."""
    options = ort.SessionOptions()
    options.graph_optimization_level = OPTIMIZATION
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


def run_ort(model: Path, inputs: Path, out: Path) -> list[str]:
    """Runs `model` on the input set in `inputs` and writes each output as
    `output_k.pb` in `out`. Returns the output names."""
    session = open_session(model)
    names = [graph_input.name for graph_input in session.get_inputs()]
    tensors = read_set(inputs)
    if len(tensors) != len(names):
        raise ValueError(f"{inputs} holds {len(tensors)} inputs, but {model} takes {len(names)}")
    outputs = session.run(None, dict(zip(names, tensors, strict=True)))
    out.mkdir(parents=True, exist_ok=True)
    for index, output in enumerate(outputs):
        (out / f"output_{index}.pb").write_bytes(
            numpy_helper.from_array(output).SerializeToString()
        )
    return [graph_output.name for graph_output in session.get_outputs()]


def cache(model: Path, inputs: Path) -> Path:
    """Computes and caches ONNX Runtime's outputs, with their provenance."""
    out = reference_dir(model, inputs)
    output_names = run_ort(model, inputs, out)
    manifest = {
        "tool": "onnxruntime",
        "version": ort.__version__,
        "provider": "CPUExecutionProvider",
        "graph_optimization": OPTIMIZATION.name,
        "device": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "command": sys.argv,
        "model": str(model),
        "model_sha256": model_sha256(model),
        "inputs": str(inputs),
        "input_set": input_set_id(inputs),
        "outputs": output_names,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def cache_cases(directory: Path) -> tuple[int, list[str]]:
    """Caches ONNX Runtime's outputs for every exported fuzz case under
    `directory`. Returns how many were cached, and each case ONNX Runtime has
    no kernel for, with its reason. Any other failure is raised."""
    cached = 0
    without_kernel = []
    for case in sorted(path for path in directory.iterdir() if path.is_dir()):
        try:
            cache(case / CASE_MODEL, case / CASE_INPUTS)
        except NoKernel as error:
            without_kernel.append(f"{case.name}: {error}")
            continue
        cached += 1
    return cached, without_kernel
