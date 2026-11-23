"""Input sets: their ids, the write/read round trip, and the input recipes."""

from pathlib import Path

import numpy as np
import pytest

from ptxc_baselines import inputs
from ptxc_baselines.paths import ROOT

#: The shared fixture's id, which `ptxc-testkit`'s tests also check.
FIXTURE = ROOT / "crates" / "testkit" / "tests" / "fixtures" / "input-set"
FIXTURE_ID = "1b45c6b27a40be0bd568351f3ad0c40074fa259d4d14b810edc0abfd59e87379"


def test_the_shared_fixture_has_its_committed_id():
    assert inputs.input_set_id(FIXTURE) == FIXTURE_ID


def test_a_written_set_reads_back_with_its_types_and_records_its_id(tmp_path):
    tensors = [np.arange(6, dtype=np.float32).reshape(2, 3), np.array([3, 1], dtype=np.int64)]

    set_id = inputs.write_set(tmp_path, tensors, {"kind": "test"})
    read = inputs.read_set(tmp_path)

    assert [tensor.dtype for tensor in read] == [np.float32, np.int64]
    for written, back in zip(tensors, read, strict=True):
        np.testing.assert_array_equal(written, back)
    assert set_id == inputs.input_set_id(tmp_path)
    assert f'"id": "{set_id}"' in (tmp_path / inputs.DESCRIPTION).read_text()


def test_the_id_changes_with_any_element_and_with_input_order(tmp_path):
    first = np.zeros(3, dtype=np.float32)
    second = np.ones(3, dtype=np.float32)
    changed = first.copy()
    changed[2] = 1e-30

    ids = [
        inputs.write_set(tmp_path / name, tensors, {})
        for name, tensors in [
            ("a", [first, second]),
            ("b", [changed, second]),
            ("c", [second, first]),
        ]
    ]

    assert len(set(ids)) == 3


def test_a_directory_without_inputs_has_no_id(tmp_path):
    with pytest.raises(FileNotFoundError):
        inputs.input_set_id(tmp_path)


@pytest.mark.parametrize(
    ("model", "vocabulary"),
    [("bert-base", inputs.BERT_VOCABULARY), ("gpt2", inputs.GPT2_VOCABULARY)],
)
def test_random_token_ids_stay_within_the_vocabulary(model, vocabulary):
    ids = inputs.random_inputs(model, batch=4, sequence=512, seed=0)[0]

    assert ids.dtype == np.int64
    assert ids.shape == (4, 512)
    assert ids.min() >= 0 and ids.max() < vocabulary


def test_bert_masks_attend_everywhere_with_one_token_type():
    _, mask, token_types = inputs.random_inputs("bert-base", batch=2, sequence=8, seed=0)

    assert (mask == 1).all() and (token_types == 0).all()


def test_random_inputs_depend_only_on_the_seed():
    def draw(seed):
        return inputs.random_inputs("resnet50", batch=1, sequence=0, seed=seed)[0]

    np.testing.assert_array_equal(draw(3), draw(3))
    assert not np.array_equal(draw(3), draw(4))
    assert draw(3).shape == (1, 3, 224, 224)


def test_a_model_without_a_recipe_is_an_error():
    with pytest.raises(ValueError, match="no input recipe"):
        inputs.random_inputs("vgg", batch=1, sequence=1, seed=0)


def make_classes(root: Path, classes: int, per_class: int):
    for label in range(classes):
        (root / str(label)).mkdir()
        for image in range(per_class):
            (root / str(label) / f"{image}.jpeg").touch()


def test_imagenet_v2_images_are_evenly_spaced_in_numeric_class_order(tmp_path):
    make_classes(tmp_path, classes=12, per_class=2)

    names = inputs.imagenet_v2_names(tmp_path, count=4)

    assert names == ["0/0.jpeg", "3/0.jpeg", "6/0.jpeg", "9/0.jpeg"]


def test_too_few_imagenet_v2_images_is_an_error(tmp_path):
    make_classes(tmp_path, classes=1, per_class=2)

    with pytest.raises(ValueError, match="fewer than 3"):
        inputs.imagenet_v2_names(tmp_path, count=3)
