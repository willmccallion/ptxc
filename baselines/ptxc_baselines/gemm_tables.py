"""The GEMM sweep's tables: ptxc against cuBLAS's default algorithm and the
best of its top 8, by type and layout, and the M3 exit's metric."""

from dataclasses import dataclass

from .tables import NONE, Results, Timing, geomean, microseconds, percent, speed, table

#: The baselines' framework names.
DEFAULT, BEST = "cublas", "cublas-best-of-8"

#: The exit metric's problems are f16 ones with M, N and K at least this.
EXIT_SIZE = 1024


@dataclass(frozen=True)
class Row:
    """One problem: its parameters and each framework's timing."""

    case: str
    params: dict
    ptxc: Timing
    default: Timing | None
    best: Timing | None

    @property
    def group(self) -> tuple[str, str]:
        return self.params["dtype"], self.params["layout"]

    def rate(self) -> float:
        """ptxc's throughput in TFLOP/s, or TOP/s for int8."""
        operations = 2 * self.params["m"] * self.params["n"] * self.params["k"]
        return operations / (self.ptxc.median_us * 1e6)


def rows(loaded: dict[str, Results]) -> list[Row]:
    """Every problem ptxc ran, in its file's order."""
    ptxc = loaded["ptxc"]
    default, best = loaded.get(DEFAULT), loaded.get(BEST)

    def baseline(results: Results | None, case: str) -> Timing | None:
        return None if results is None else results.timings.get(case)

    return [
        Row(case, ptxc.params[case], timing, baseline(default, case), baseline(best, case))
        for case, timing in ptxc.timings.items()
    ]


def size(params: dict) -> str:
    """A problem's M, N and K, as its id has them."""
    return f"{params['m']}x{params['n']}x{params['k']}"


def groups(found: list[Row]) -> dict[tuple[str, str], list[Row]]:
    """`found` by type and layout, in order of first appearance."""
    grouped: dict[tuple[str, str], list[Row]] = {}
    for row in found:
        grouped.setdefault(row.group, []).append(row)
    return grouped


def ratios(found: list[Row], baseline: str) -> list[float]:
    """ptxc's speed against `baseline` on each problem that has one."""
    timings = [speed(row.ptxc, getattr(row, baseline)) for row in found]
    return [ratio for ratio in timings if ratio is not None]


def slowest(found: list[Row]) -> str:
    """The problem ptxc runs slowest on against cuBLAS's default, and how."""
    against = [(speed(row.ptxc, row.default), row) for row in found if row.default]
    if not against:
        return NONE
    ratio, row = min(against, key=lambda pair: pair[0])
    return f"{size(row.params)} ({percent(ratio)})"


def summary_row(group: tuple[str, str], found: list[Row]) -> list[str]:
    against = ratios(found, "default")
    faster = sum(1 for ratio in against if ratio >= 1)
    return [
        *group,
        str(len(found)),
        percent(geomean(against)),
        percent(geomean(ratios(found, "best"))),
        f"{faster} of {len(against)}",
        slowest(found),
    ]


def exit_metric(found: list[Row]) -> str:
    """ptxc's geometric-mean speed against cuBLAS on the f16 problems with
    M, N and K at least `EXIT_SIZE`, by layout and over all of them."""
    large = [
        row
        for row in found
        if row.params["dtype"] == "f16"
        and min(row.params["m"], row.params["n"], row.params["k"]) >= EXIT_SIZE
    ]
    layouts = groups(large)
    parts = [
        f"{layout} {percent(geomean(ratios(rows, 'default')))}"
        for (_, layout), rows in layouts.items()
    ]
    overall = percent(geomean(ratios(large, "default")))
    return (
        f"f16 GEMMs with M, N and K at least {EXIT_SIZE}, geometric mean of cuBLAS's "
        f"throughput: {overall} ({', '.join(parts)})."
    )


def summary(found: list[Row]) -> str:
    """By type and layout: ptxc's geometric-mean percentage of cuBLAS's
    throughput, against its default and its best of 8, the problems it
    wins, and its slowest; then the exit metric."""
    header = [
        "Type",
        "Layout",
        "Problems",
        "% of cuBLAS",
        "% of best of 8",
        "Faster than cuBLAS",
        "Slowest against cuBLAS",
    ]
    body = [summary_row(group, rows) for group, rows in groups(found).items()]
    return table(header, body) + "\n\n" + exit_metric(found)


def full(found: list[Row]) -> str:
    """Every problem, a section per type and layout."""
    header = ["Problem", "ptxc µs", "T(FL)OP/s", "cuBLAS µs", "%", "best of 8 µs", "%"]
    sections = []
    for (dtype, layout), rows in groups(found).items():
        body = [
            [
                size(row.params),
                microseconds(row.ptxc),
                f"{row.rate():.2f}",
                microseconds(row.default),
                percent(speed(row.ptxc, row.default)),
                microseconds(row.best),
                percent(speed(row.ptxc, row.best)),
            ]
            for row in rows
        ]
        sections.append(f"## {dtype} {layout}\n\n" + table(header, body))
    return "\n\n".join(sections)
