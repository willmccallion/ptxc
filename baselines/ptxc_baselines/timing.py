"""Times a launch with CUDA events, as BENCHMARKING.md's methodology says
and as `ptxc-bench` does (crates/bench/src/timing.rs): a warmup of at least
20 iterations and 1 s, then a series of at least 100 samples and 2 s. A
sample is a batch of back-to-back launches between two events on one
stream, sized in the warmup to last at least 1 ms; it records the time per
launch."""

import time as clock
from collections.abc import Callable
from dataclasses import dataclass

#: How long a sample lasts at the least, in microseconds.
SAMPLE_TIME_US = 1000.0

#: The most launches a sample batches, however short they are.
MAX_BATCH = 100_000


@dataclass(frozen=True)
class Budget:
    """How many iterations, and for how long, a phase runs at the least."""

    iterations: int
    seconds: float

    def is_met(self, iterations: int, elapsed: float) -> bool:
        return iterations >= self.iterations and elapsed >= self.seconds


WARMUP = Budget(20, 1.0)
TIMED = Budget(100, 2.0)


@dataclass(frozen=True)
class Series:
    """Each sample's time per launch in microseconds, and the launches each
    sample batched."""

    samples_us: list[float]
    batch: int


def percentile(ordered: list[float], percent: int) -> float:
    """The `percent`th percentile of `ordered`, between the two nearest
    ranks, as numpy's default percentile is."""
    if not ordered:
        raise ValueError("a percentile of no samples")
    index, remainder = divmod((len(ordered) - 1) * percent, 100)
    lower = ordered[index]
    if remainder == 0:
        return lower
    return (ordered[index + 1] - lower) * (remainder / 100) + lower


def summary(series: Series) -> tuple[float, float, float]:
    """The median, p10 and p90 of `series`."""
    ordered = sorted(series.samples_us)
    return percentile(ordered, 50), percentile(ordered, 10), percentile(ordered, 90)


def batch_size(warmup: Series) -> int:
    """The launches a sample batches to last `SAMPLE_TIME_US`, from the
    warmup's median: the smallest power of two that does, at most
    `MAX_BATCH`."""
    per_launch = summary(warmup)[0]
    if per_launch >= SAMPLE_TIME_US:
        return 1
    needed = SAMPLE_TIME_US / max(per_launch, 1e-300)
    batch = 1
    while batch < MAX_BATCH and batch < needed:
        batch = min(batch * 2, MAX_BATCH)
    return batch


def run(budget: Budget, iteration: Callable[[], float]) -> list[float]:
    """Each time `iteration` gives, run until `budget` is met."""
    began = clock.monotonic()
    samples: list[float] = []
    while not budget.is_met(len(samples), clock.monotonic() - began):
        samples.append(iteration())
    return samples


def time(launch: Callable[[], None]) -> Series:
    """Times `launch`, which queues one iteration's work on the current CUDA
    stream: a warmup of single launches, then a timed series of batches
    sized from it."""
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def sample(batch: int) -> float:
        start.record()
        for _ in range(batch):
            launch()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0 / batch

    warmup = Series(run(WARMUP, lambda: sample(1)), 1)
    batch = batch_size(warmup)
    return Series(run(TIMED, lambda: sample(batch)), batch)
