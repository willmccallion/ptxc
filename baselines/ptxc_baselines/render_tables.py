"""`ptxc-baselines render`: the README's result tables from one run's
results, `bench/results/<run>/`, as BENCHMARKING.md describes (D77).

The README gets a summary per suite between its `<!-- bench:<suite>:begin
-->` and `<!-- bench:<suite>:end -->` markers, which links to `<suite>.md`
beside the results, where every case is. A suite with no ptxc results in
the run is left as it was. Results from uncommitted code, a case that was
not verified, and a series whose clocks moved are refused."""

import os
from collections.abc import Callable
from pathlib import Path

from . import conv_tables, gemm_tables
from .tables import RenderError, Results, capped_note, load, replace_between, write_full

__all__ = ["RenderError", "render"]

#: Each suite: its title, and how its summary and full tables are made.
SUITES: dict[str, tuple[str, Callable[[dict[str, Results]], list], Callable, Callable]] = {
    "gemm": ("GEMM against cuBLAS", gemm_tables.rows, gemm_tables.summary, gemm_tables.full),
    "conv": (
        "ResNet-50's convolutions against cuDNN",
        conv_tables.rows,
        conv_tables.summary,
        conv_tables.full,
    ),
}


def render(directory: Path, readme: Path) -> list[str]:
    """Rewrites `readme`'s tables and writes each suite's full tables from
    the results in `directory`; the suites rendered."""
    text = readme.read_text()
    rendered = []
    for suite, (title, rows, summary, full) in SUITES.items():
        loaded = load(directory, suite)
        if "ptxc" not in loaded:
            continue
        found = rows(loaded)
        page = directory / f"{suite}.md"
        write_full(page, title, loaded, full(found))
        link = Path(os.path.relpath(page, readme.parent)).as_posix()
        body = f"{summary(found)}{capped_note(loaded)}\n\nEvery case: [{link}]({link})."
        text = replace_between(text, suite, body)
        rendered.append(suite)
    readme.write_text(text)
    return rendered
