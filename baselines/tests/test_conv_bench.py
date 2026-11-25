"""The convolution baseline's problems and its check, on the CPU."""

import torch
import torch.nn.functional as functional

from ptxc_baselines import conv_bench


def test_the_suite_has_resnet50s_23_layers_at_three_batch_sizes():
    problems = conv_bench.problems()

    ids = {problem.id for problem in problems}

    assert len(problems) == 23 * 3 == len(ids)
    assert "conv/f16/stem/b32" in ids
    stem = next(problem for problem in problems if problem.layer.name == "stem")
    assert (stem.layer.output, stem.layer.depth) == (112, 147)


def test_a_result_records_the_parameters_ptxc_bench_does():
    problem = conv_bench.problems()[0]

    assert list(problem.params()) == [
        "layer",
        "batch",
        "channels",
        "features",
        "size",
        "kernel",
        "stride",
        "pad",
        "dtype",
    ]
    assert problem.params()["layer"] == "stem"


def test_samples_spread_to_the_last_element_or_take_every_one():
    assert conv_bench.positions(10) == list(range(10))
    spread = conv_bench.positions(100_000)
    assert len(spread) == conv_bench.SAMPLES
    assert spread[0] == 0 and spread[-1] > 99_000


def tiny() -> conv_bench.Problem:
    layer = conv_bench.Layer("tiny", channels=2, features=3, size=5, kernel=3, stride=2, pad=1)
    return conv_bench.Problem(layer, batch=2, dtype="f16")


def test_operands_are_eighths_in_the_unit_range():
    x, w = conv_bench.operands(tiny())

    assert (x.shape, w.shape) == ((2, 2, 5, 5), (3, 2, 3, 3))
    for values in (x, w):
        assert bool(((values * 8).frac() == 0).all()) and float(values.abs().max()) <= 1


def test_the_true_convolution_passes_and_a_wrong_element_fails():
    problem = tiny()
    x, w = conv_bench.operands(problem)
    expected = functional.conv2d(x, w, stride=2, padding=1)
    computed = functional.conv2d(x.half().float(), w.half().float(), stride=2, padding=1).half()
    wrong = computed.clone()
    wrong[1, 2, 2, 2] += 1

    assert conv_bench.check(problem, computed, expected) is None
    assert "1 of 54 elements differ" in conv_bench.check(problem, wrong, expected)


def test_a_channels_last_output_is_checked_in_its_logical_order():
    problem = tiny()
    x, w = conv_bench.operands(problem)
    expected = functional.conv2d(x, w, stride=2, padding=1)
    channels_last = expected.half().contiguous(memory_format=torch.channels_last)

    assert conv_bench.check(problem, channels_last, expected) is None


def test_a_nan_fails_the_check():
    problem = tiny()
    x, w = conv_bench.operands(problem)
    expected = functional.conv2d(x, w, stride=2, padding=1)
    unwritten = torch.full_like(expected, float("nan")).half()

    assert conv_bench.check(problem, unwritten, expected) is not None
