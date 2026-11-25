"""Reading `nvidia-smi`'s answers, as ptxc-bench reads them."""

import pytest

from ptxc_baselines import gpu

#: `nvidia-smi`'s answer on an idle RTX 4060 Ti.
IDLE = "210, 405, 45, Active, " + ", ".join(["Not Active"] * 7) + "\n"


def test_a_state_line_gives_the_clocks_temperature_and_active_reasons():
    assert gpu.parse_state(IDLE) == gpu.GpuState(210, 405, 45, ["gpu_idle"])


def test_every_reason_is_read_in_its_place():
    reasons = ["Not Active", "Active", "Not Active", "Not Active", "Active", "Not Active"]
    line = "2520, 9001, 71, " + ", ".join([*reasons, "Not Active", "Active"])

    assert gpu.parse_state(line).events == ["sw_power_cap", "hw_thermal_slowdown", "sync_boost"]


def test_a_malformed_state_line_is_refused():
    with pytest.raises(gpu.GpuError):
        gpu.parse_state("2520, 9001")
    with pytest.raises(gpu.GpuError):
        gpu.parse_state(IDLE.replace("Not Active", "Unknown", 1))
    with pytest.raises(gpu.GpuError):
        gpu.parse_state(IDLE.replace("210", "N/A"))


def test_clocks_hold_only_where_the_sm_and_memory_clocks_did_not_move():
    before = gpu.GpuState(2700, 8751, 50, [])

    assert gpu.clocks_held(before, gpu.GpuState(2700, 8751, 66, ["sw_power_cap"]))
    assert not gpu.clocks_held(before, gpu.GpuState(2460, 8751, 66, []))


def test_other_processes_leave_out_this_one():
    listed = "4242, /usr/bin/python3\n17, ptxc-bench\n"

    assert gpu.other_processes(listed, own=4242) == [(17, "ptxc-bench")]
    assert gpu.other_processes("", own=1) == []
    with pytest.raises(gpu.GpuError):
        gpu.other_processes("not a process", own=1)
