"""A suite's results as JSON, in BENCHMARKING.md's schema, as `ptxc-bench`
writes them (crates/bench/src/results.rs): who ran it and where, and each
case's samples and summary, with the GPU's clocks around its series."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import gpu, timing

#: The schema version these results follow.
SCHEMA = 1


@dataclass(frozen=True)
class Framework:
    """The framework timed, its version, and the commit of the harness."""

    name: str
    version: str
    commit: str
    dirty: bool

    def as_json(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "commit": self.commit,
            "dirty": self.dirty,
        }


def git(repository: Path, *arguments: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repository), *arguments], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)}: {done.stderr.strip()}")
    return done.stdout


def framework(name: str, version: str, repository: Path) -> Framework:
    """`name` at `version`, run from the harness at `repository`'s commit."""
    commit = git(repository, "rev-parse", "--short=12", "HEAD").strip()
    dirty = bool(git(repository, "status", "--porcelain").strip())
    return Framework(name, version, commit, dirty)


def case(
    case_id: str,
    params: dict,
    clocks: tuple[gpu.GpuState, gpu.GpuState],
    series: timing.Series,
) -> dict:
    """A verified case from its timed `series` and the GPU's state around it."""
    median, p10, p90 = timing.summary(series)
    before, after = clocks
    return {
        "id": case_id,
        "params": params,
        "verified": True,
        "clocks": {
            "before": before.as_json(),
            "after": after.as_json(),
            "held": gpu.clocks_held(before, after),
        },
        "iterations_per_sample": series.batch,
        "samples_us": list(series.samples_us),
        "median_us": median,
        "p10_us": p10,
        "p90_us": p90,
    }


def write(
    path: Path,
    suite: str,
    framework: Framework,
    environment: tuple[str, str],
    cases: list[dict],
) -> None:
    """The results of `suite` for `framework`, run on the GPU and driver of
    `environment`, written to `path` as indented JSON."""
    name, driver = environment
    results = {
        "schema": SCHEMA,
        "suite": suite,
        "framework": framework.as_json(),
        "environment": {"gpu": name, "driver": driver},
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
