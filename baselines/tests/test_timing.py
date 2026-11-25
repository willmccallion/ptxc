"""The timing methodology's statistics, as ptxc-bench computes them."""

import pytest

from ptxc_baselines import timing


def test_percentiles_interpolate_between_the_nearest_ranks_as_numpy_does():
    ordered = [1.0, 2.0, 3.0, 4.0]

    assert timing.percentile(ordered, 50) == 2.5
    assert timing.percentile(ordered, 10) == pytest.approx(1.3)
    assert timing.percentile(ordered, 90) == pytest.approx(3.7)
    assert timing.percentile([7.0], 90) == 7.0


def test_no_samples_have_no_percentile():
    with pytest.raises(ValueError):
        timing.percentile([], 50)


def test_a_series_summary_is_its_median_p10_and_p90_whatever_its_order():
    series = timing.Series([4.0, 1.0, 3.0, 2.0], batch=8)

    median, p10, p90 = timing.summary(series)

    assert (median, p10, p90) == (2.5, pytest.approx(1.3), pytest.approx(3.7))


def batch(per_launch: float) -> int:
    """The batch a warmup of launches lasting `per_launch` microseconds gives."""
    return timing.batch_size(timing.Series([per_launch] * 5, 1))


def test_a_sample_batches_the_smallest_power_of_two_lasting_a_millisecond():
    assert batch(2000.0) == 1
    assert batch(1000.0) == 1
    assert batch(300.0) == 4
    assert batch(250.0) == 4
    assert batch(1e-9) == timing.MAX_BATCH


def test_a_budget_needs_both_its_iterations_and_its_time():
    budget = timing.Budget(iterations=20, seconds=1.0)

    assert budget.is_met(20, 1.0)
    assert not budget.is_met(19, 5.0)
    assert not budget.is_met(500, 0.5)
