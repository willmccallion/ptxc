"""What every suite's tables share: results read from a run's files and
checked for publishing, joined on case id, and written as Markdown."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

#: The results schema this reads.
SCHEMA = 1

#: The mark of a series during which the GPU reported a power or thermal
#: cap, though its clocks held.
CAPPED = "†"

#: A cell with no value: an en dash.
NONE = "\u2013"


class RenderError(Exception):
    """Results that may not be published, or a README without the markers."""


@dataclass(frozen=True)
class Timing:
    """A case's time per run, in microseconds, and whether the GPU reported
    a cap during its series."""

    median_us: float
    p10_us: float
    p90_us: float
    capped: bool


@dataclass(frozen=True)
class Results:
    """One framework's file: who ran it, where, and each case's parameters
    and timing by id, in the file's order."""

    framework: str
    version: str
    commit: str
    environment: dict
    params: dict[str, dict]
    timings: dict[str, Timing]
    sm_mhz: frozenset[int]


def problems(document: dict, name: str) -> list[str]:
    """Why the results `document`, read from `name`, may not be published."""
    found = []
    if document.get("schema") != SCHEMA:
        found.append(f"{name}: schema {document.get('schema')}, not {SCHEMA}")
    if document["framework"]["dirty"]:
        found.append(f"{name}: run from uncommitted changes")
    for case in document["cases"]:
        if not case["verified"]:
            found.append(f"{name}: {case['id']} was not verified")
        if not case["clocks"]["held"]:
            found.append(f"{name}: {case['id']}'s clocks moved during its series")
    return found


def results(document: dict) -> Results:
    """The published parts of a checked results `document`."""
    framework = document["framework"]
    timings = {}
    clocks = set()
    for case in document["cases"]:
        before, after = case["clocks"]["before"], case["clocks"]["after"]
        capped = bool(before["events"] or after["events"])
        timings[case["id"]] = Timing(case["median_us"], case["p10_us"], case["p90_us"], capped)
        clocks.add(before["sm_mhz"])
    return Results(
        framework=framework["name"],
        version=framework["version"],
        commit=framework["commit"],
        environment=document["environment"],
        params={case["id"]: case["params"] for case in document["cases"]},
        timings=timings,
        sm_mhz=frozenset(clocks),
    )


def load(directory: Path, suite: str) -> dict[str, Results]:
    """Every framework's results for `suite` in `directory`, by framework
    name, after refusing any that may not be published."""
    documents = {
        path.name: json.loads(path.read_text())
        for path in sorted(directory.glob(f"{suite}-*.json"))
    }
    found = [
        problem for name, document in documents.items() for problem in problems(document, name)
    ]
    if found:
        raise RenderError("results that may not be published:\n" + "\n".join(found))
    loaded = [results(document) for document in documents.values()]
    return {result.framework: result for result in loaded}


def geomean(ratios: list[float]) -> float | None:
    """The geometric mean of `ratios`, or `None` if there are none."""
    if not ratios:
        return None
    return math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))


def percent(ratio: float | None) -> str:
    return NONE if ratio is None else f"{100 * ratio:.0f}%"


def microseconds(timing: Timing | None) -> str:
    """`timing`'s median, marked if the GPU reported a cap during it."""
    if timing is None:
        return NONE
    mark = CAPPED if timing.capped else ""
    return f"{timing.median_us:,.1f}{mark}"


def speed(ours: Timing, theirs: Timing | None) -> float | None:
    """How fast `ours` runs against `theirs`, 1 for the same time."""
    return None if theirs is None else theirs.median_us / ours.median_us


def table(header: list[str], rows: list[list[str]]) -> str:
    """A Markdown table, columns after the first aligned right."""
    rule = ["---"] + ["---:"] * (len(header) - 1)
    lines = [header, rule, *rows]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in lines)


def replace_between(text: str, suite: str, body: str) -> str:
    """`text` with what lies between `suite`'s markers replaced by `body`."""
    begin, end = f"<!-- bench:{suite}:begin -->", f"<!-- bench:{suite}:end -->"
    start, stop = text.find(begin), text.find(end)
    if start < 0 or stop < start:
        raise RenderError(f"the README has no {begin} ... {end} markers")
    return text[: start + len(begin)] + "\n" + body + "\n" + text[stop:]


def provenance(loaded: dict[str, Results]) -> str:
    """Where and how the results were taken: GPU, driver, SM clock, and
    each framework's version and commit."""
    first = next(iter(loaded.values()))
    clocks = sorted(frozenset().union(*(result.sm_mhz for result in loaded.values())))
    clock = ", ".join(f"{mhz} MHz" for mhz in clocks)
    lines = [
        f"{first.environment['gpu']}, driver {first.environment['driver']}, SM at {clock}.",
        "",
    ]
    for result in loaded.values():
        lines.append(f"- {result.framework} {result.version}, commit `{result.commit}`")
    return "\n".join(lines)


def capped_note(loaded: dict[str, Results]) -> str:
    """A note on the marked series, if there are any."""
    if not any(timing.capped for result in loaded.values() for timing in result.timings.values()):
        return ""
    return (
        f"\n\n{CAPPED} The GPU reported a power or thermal cap during the series; its clocks held."
    )


def write_full(path: Path, title: str, loaded: dict[str, Results], body: str) -> None:
    """`body`, every case's table, under `title` and the run's provenance."""
    text = f"# {title}\n\n{provenance(loaded)}\n\n{body}{capped_note(loaded)}\n"
    path.write_text(text)
