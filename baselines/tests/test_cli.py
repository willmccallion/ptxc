"""The command line: argument parsing and image selection."""

import pytest

from ptxc_baselines import cli, paths


def test_imagenet_takes_the_first_images_of_every_tenth_in_the_list():
    names = cli.select_images("imagenet", paths.IMAGENET_VAL, count=3)

    assert names == [
        "ILSVRC2012_val_00000010.JPEG",
        "ILSVRC2012_val_00000020.JPEG",
        "ILSVRC2012_val_00000030.JPEG",
    ]


def test_the_imagenet_list_holds_5000_distinct_images():
    names = paths.IMAGENET_LIST.read_text().split()

    assert len(set(names)) == 5000
    assert names[-1] == "ILSVRC2012_val_00050000.JPEG"


def test_random_sets_default_to_one_sequence_of_128():
    args = cli.parse(["random", "--model", "gpt2"])

    assert (args.batch, args.sequence, args.seed) == (1, 128, 0)


def test_an_unknown_image_source_is_rejected():
    with pytest.raises(SystemExit):
        cli.parse(["images", "--source", "cifar"])


def test_a_missing_image_directory_says_how_to_get_it(monkeypatch, tmp_path):
    missing = tmp_path / "absent"
    monkeypatch.setitem(cli.IMAGE_SOURCES, "imagenetv2", (missing, "imagenetv2", "fetch it"))

    with pytest.raises(SystemExit, match="absent does not exist: fetch it"):
        cli.run_images(cli.parse(["images", "--source", "imagenetv2"]))
