"""Results files in BENCHMARKING.md's schema."""

import json
import subprocess

from ptxc_baselines import gpu, results, timing


def test_a_case_records_its_summary_samples_and_clocks():
    before, after = gpu.GpuState(2700, 8751, 50, []), gpu.GpuState(2700, 8751, 61, [])
    series = timing.Series([3.0, 1.0, 2.0], batch=16)

    case = results.case("conv/f16/stem/b1", {"batch": 1}, (before, after), series)

    assert case["verified"] and case["clocks"]["held"]
    assert (case["median_us"], case["iterations_per_sample"]) == (2.0, 16)
    assert case["samples_us"] == [3.0, 1.0, 2.0]
    assert case["clocks"]["after"]["temperature_c"] == 61


def test_results_are_written_in_the_schema(tmp_path):
    framework = results.Framework("pytorch", "2.14.1", "abc123def456", dirty=False)

    path = tmp_path / "out" / "conv-pytorch.json"
    results.write(path, "conv", framework, ("NVIDIA GeForce RTX 4060 Ti", "595.112.07"), [])

    written = json.loads(path.read_text())
    assert written == {
        "schema": 1,
        "suite": "conv",
        "framework": {
            "name": "pytorch",
            "version": "2.14.1",
            "commit": "abc123def456",
            "dirty": False,
        },
        "environment": {"gpu": "NVIDIA GeForce RTX 4060 Ti", "driver": "595.112.07"},
        "cases": [],
    }


def test_a_framework_records_its_harnesss_commit_and_whether_it_is_dirty(tmp_path):
    git = ["git", "-C", str(tmp_path), "-c", "core.hooksPath=/dev/null"]
    git += ["-c", "commit.gpgsign=false"]
    identity = ["-c", "user.name=bench", "-c", "user.email=bench@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, *identity, "commit", "-q", "--allow-empty", "-m", "empty"], check=True)

    clean = results.framework("pytorch", "2.14.1", tmp_path)
    (tmp_path / "file").write_text("change")
    dirty = results.framework("pytorch", "2.14.1", tmp_path)

    assert len(clean.commit) == 12 and not clean.dirty
    assert dirty.dirty and dirty.commit == clean.commit
