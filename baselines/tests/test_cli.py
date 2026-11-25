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


def test_render_defaults_to_the_repositorys_readme():
    args = cli.parse(["render", "--results", "bench/results/run"])

    assert args.readme == str(paths.ROOT / "README.md")


def test_results_that_may_not_be_published_end_the_render_with_why(tmp_path):
    (tmp_path / "gemm-ptxc.json").write_text(
        '{"schema": 1, "framework": {"name": "ptxc", "version": "0.1.0", '
        '"commit": "abc", "dirty": true}, "environment": {}, "cases": []}'
    )
    readme = tmp_path / "README.md"
    readme.write_text("unchanged")

    with pytest.raises(SystemExit, match=r"gemm-ptxc\.json: run from uncommitted changes"):
        cli.run_render(cli.parse(["render", "--results", str(tmp_path), "--readme", str(readme)]))
    assert readme.read_text() == "unchanged"
