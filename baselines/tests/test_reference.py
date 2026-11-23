"""ONNX Runtime references: outputs, the cache layout, and the manifest."""

import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from ptxc_baselines import inputs, reference


def add_one_model(path):
    """`y = x + 1` on a float vector of length 3."""
    one = numpy_helper.from_array(np.ones(3, dtype=np.float32), name="one")
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "one"], ["y"])],
        "add_one",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
        [one],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)]), path)
    return path


def read_output(directory, index):
    return numpy_helper.to_array(onnx.load_tensor(str(directory / f"output_{index}.pb")))


def test_sessions_run_the_model_as_written_on_the_cpu(tmp_path):
    session = reference.open_session(add_one_model(tmp_path / "model.onnx"))

    level = session.get_session_options().graph_optimization_level
    assert level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_outputs_are_written_in_order_as_tensors(tmp_path):
    model = add_one_model(tmp_path / "model.onnx")
    inputs.write_set(tmp_path / "set", [np.array([1, 2, 3], dtype=np.float32)], {})

    names = reference.run_ort(model, tmp_path / "set", tmp_path / "out")

    assert names == ["y"]
    np.testing.assert_array_equal(read_output(tmp_path / "out", 0), [2, 3, 4])


def test_a_set_with_the_wrong_number_of_inputs_is_an_error(tmp_path):
    model = add_one_model(tmp_path / "model.onnx")
    extra = [np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]
    inputs.write_set(tmp_path / "set", extra, {})

    with pytest.raises(ValueError, match="holds 2 inputs"):
        reference.run_ort(model, tmp_path / "set", tmp_path / "out")


def test_the_cache_is_keyed_by_model_input_set_and_tool_version(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "REFERENCE", tmp_path / "reference")
    model = add_one_model(tmp_path / "model.onnx")
    set_id = inputs.write_set(tmp_path / "set", [np.zeros(3, dtype=np.float32)], {})

    out = reference.cache(model, tmp_path / "set")

    model_sha = reference.model_sha256(model)
    assert out == tmp_path / "reference" / model_sha / set_id / f"ort-{ort.__version__}"
    manifest = json.loads((out / reference.MANIFEST).read_text())
    assert manifest["model_sha256"] == model_sha
    assert manifest["input_set"] == set_id
    assert manifest["provider"] == "CPUExecutionProvider"
    assert manifest["graph_optimization"] == "ORT_DISABLE_ALL"
    assert manifest["outputs"] == ["y"]


def fuzz_case(directory, dtype, x):
    """An exported fuzz case: `y = relu(x)` on `x`, of ONNX type `dtype`."""
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])],
        "case",
        [helper.make_tensor_value_info("x", dtype, list(x.shape))],
        [helper.make_tensor_value_info("y", dtype, list(x.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    directory.mkdir(parents=True)
    onnx.save(model, directory / reference.CASE_MODEL)
    inputs.write_set(directory / reference.CASE_INPUTS, [x], {"case": directory.name})


def test_every_fuzz_case_is_cached_and_cases_without_a_kernel_are_listed(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "REFERENCE", tmp_path / "reference")
    cases = tmp_path / "cases"
    fuzz_case(cases / "seed-1", TensorProto.FLOAT, np.array([-1.0, 2.0], dtype=np.float32))
    bf16 = numpy_helper.to_array(helper.make_tensor("x", TensorProto.BFLOAT16, [2], [1.0, -2.0]))
    fuzz_case(cases / "seed-2", TensorProto.BFLOAT16, bf16)

    cached, without_kernel = reference.cache_cases(cases)

    assert cached == 1
    assert [reason.split(":")[0] for reason in without_kernel] == ["seed-2"]
    model, set_dir = (
        cases / "seed-1" / reference.CASE_MODEL,
        cases / "seed-1" / reference.CASE_INPUTS,
    )
    np.testing.assert_array_equal(
        read_output(reference.reference_dir(model, set_dir), 0), [0.0, 2.0]
    )
