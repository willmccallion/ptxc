"""The convolution sweep's tables: ptxc against PyTorch with cuDNN, in NCHW
as ptxc takes a layer and in channels-last, by batch size."""

from dataclasses import dataclass

from .tables import NONE, Results, Timing, geomean, microseconds, percent, speed, table

#: The baselines' framework names: NCHW, and channels-last.
NCHW, CHANNELS_LAST = "pytorch", "pytorch-channels-last"


@dataclass(frozen=True)
class Row:
    """One layer at one batch size: its parameters and each framework's
    timing."""

    case: str
    params: dict
    ptxc: Timing
    nchw: Timing | None
    channels_last: Timing | None

    def rate(self) -> float:
        """ptxc's throughput in TFLOP/s, two operations per multiply-add of
        the direct convolution."""
        params = self.params
        side = (params["size"] + 2 * params["pad"] - params["kernel"]) // params["stride"] + 1
        outputs = params["batch"] * params["features"] * side * side
        operations = 2 * outputs * params["channels"] * params["kernel"] ** 2
        return operations / (self.ptxc.median_us * 1e6)


def rows(loaded: dict[str, Results]) -> list[Row]:
    """Every layer ptxc ran, in its file's order."""
    ptxc = loaded["ptxc"]
    nchw, channels_last = loaded.get(NCHW), loaded.get(CHANNELS_LAST)

    def baseline(results: Results | None, case: str) -> Timing | None:
        return None if results is None else results.timings.get(case)

    return [
        Row(
            case,
            ptxc.params[case],
            timing,
            baseline(nchw, case),
            baseline(channels_last, case),
        )
        for case, timing in ptxc.timings.items()
    ]


def batches(found: list[Row]) -> dict[int, list[Row]]:
    """`found` by batch size, in order of first appearance."""
    grouped: dict[int, list[Row]] = {}
    for row in found:
        grouped.setdefault(row.params["batch"], []).append(row)
    return grouped


def ratios(found: list[Row], baseline: str) -> list[float]:
    """ptxc's speed against `baseline` on each layer that has one."""
    timings = [speed(row.ptxc, getattr(row, baseline)) for row in found]
    return [ratio for ratio in timings if ratio is not None]


def slowest(found: list[Row]) -> str:
    """The layer ptxc runs slowest on against cuDNN in NCHW, and how."""
    against = [(speed(row.ptxc, row.nchw), row) for row in found if row.nchw]
    if not against:
        return NONE
    ratio, row = min(against, key=lambda pair: pair[0])
    return f"{row.params['layer']} ({percent(ratio)})"


def summary(found: list[Row]) -> str:
    """By batch size: ptxc's geometric-mean percentage of cuDNN's
    throughput in each layout, the layers it wins, and its slowest."""
    header = [
        "Batch",
        "Layers",
        "% of cuDNN NCHW",
        "% of cuDNN channels-last",
        "Faster than NCHW",
        "Slowest against NCHW",
    ]
    body = []
    for batch, rows in batches(found).items():
        against = ratios(rows, "nchw")
        faster = sum(1 for ratio in against if ratio >= 1)
        body.append(
            [
                str(batch),
                str(len(rows)),
                percent(geomean(against)),
                percent(geomean(ratios(rows, "channels_last"))),
                f"{faster} of {len(against)}",
                slowest(rows),
            ]
        )
    return table(header, body)


def full(found: list[Row]) -> str:
    """Every layer, a section per batch size."""
    header = ["Layer", "ptxc µs", "TFLOP/s", "NCHW µs", "%", "channels-last µs", "%"]
    sections = []
    for batch, rows in batches(found).items():
        body = [
            [
                row.params["layer"],
                microseconds(row.ptxc),
                f"{row.rate():.2f}",
                microseconds(row.nchw),
                percent(speed(row.ptxc, row.nchw)),
                microseconds(row.channels_last),
                percent(speed(row.ptxc, row.channels_last)),
            ]
            for row in rows
        ]
        sections.append(f"## Batch {batch}\n\n" + table(header, body))
    return "\n\n".join(sections)
