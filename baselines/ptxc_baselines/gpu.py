"""The GPU's state and identity from `nvidia-smi`, as `ptxc-bench` reads
them (crates/bench/src/clocks.rs), so a baseline's results record the same
fields: the clocks and temperature around each series, the reasons the
clocks are held down, and whether any other process is computing."""

import os
import subprocess
from dataclasses import dataclass

#: The GPU every suite runs on.
GPU = "0"

#: Each reason `nvidia-smi` may give for holding the clocks down, by the
#: name a result records and the field it is queried as.
EVENTS = [
    ("gpu_idle", "clocks_event_reasons.gpu_idle"),
    ("sw_power_cap", "clocks_event_reasons.sw_power_cap"),
    ("hw_slowdown", "clocks_event_reasons.hw_slowdown"),
    ("sw_thermal_slowdown", "clocks_event_reasons.sw_thermal_slowdown"),
    ("hw_thermal_slowdown", "clocks_event_reasons.hw_thermal_slowdown"),
    ("hw_power_brake_slowdown", "clocks_event_reasons.hw_power_brake_slowdown"),
    ("applications_clocks_setting", "clocks_event_reasons.applications_clocks_setting"),
    ("sync_boost", "clocks_event_reasons.sync_boost"),
]

#: The clocks and temperature, queried before the reasons.
CLOCK_FIELDS = ["clocks.sm", "clocks.mem", "temperature.gpu"]


class GpuError(RuntimeError):
    """`nvidia-smi` failed, answered unexpectedly, or the GPU is busy."""


@dataclass(frozen=True)
class GpuState:
    """The SM and memory clocks, the temperature, and the active reasons."""

    sm_mhz: int
    mem_mhz: int
    temperature_c: int
    events: list[str]

    def as_json(self) -> dict:
        return {
            "sm_mhz": self.sm_mhz,
            "mem_mhz": self.mem_mhz,
            "temperature_c": self.temperature_c,
            "events": list(self.events),
        }


def nvidia_smi(*arguments: str) -> str:
    done = subprocess.run(
        ["nvidia-smi", "-i", GPU, *arguments], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise GpuError(f"nvidia-smi failed: {done.stderr.strip()}")
    return done.stdout


def parse_state(line: str) -> GpuState:
    """A state from one `--query-gpu` line of the clock fields and reasons."""
    fields = CLOCK_FIELDS + [field for _, field in EVENTS]
    values = line.strip().split(", ")
    if len(values) != len(fields):
        raise GpuError(f"nvidia-smi gave {len(values)} fields where {len(fields)} were asked for")
    try:
        sm, memory, temperature = (int(value) for value in values[: len(CLOCK_FIELDS)])
    except ValueError as error:
        raise GpuError(f"nvidia-smi gave {values[:3]} for the clocks") from error
    events = []
    for (event, field), value in zip(EVENTS, values[len(CLOCK_FIELDS) :], strict=True):
        if value == "Active":
            events.append(event)
        elif value != "Not Active":
            raise GpuError(f"nvidia-smi gave {value!r} for {field}")
    return GpuState(sm, memory, temperature, events)


def state() -> GpuState:
    fields = ",".join(CLOCK_FIELDS + [field for _, field in EVENTS])
    return parse_state(nvidia_smi(f"--query-gpu={fields}", "--format=csv,noheader,nounits"))


def clocks_held(before: GpuState, after: GpuState) -> bool:
    """Whether a series ran at one SM and memory clock."""
    return (before.sm_mhz, before.mem_mhz) == (after.sm_mhz, after.mem_mhz)


def identity() -> tuple[str, str]:
    """The GPU's name and the driver's version."""
    line = nvidia_smi("--query-gpu=name,driver_version", "--format=csv,noheader").strip()
    name, _, driver = line.partition(", ")
    if not driver:
        raise GpuError(f"nvidia-smi gave {line!r} for name,driver_version")
    return name, driver


def other_processes(output: str, own: int) -> list[tuple[int, str]]:
    """The compute processes `--query-compute-apps` lists, but `own`."""
    others = []
    for line in filter(None, (line.strip() for line in output.splitlines())):
        pid, _, name = line.partition(", ")
        if not pid.isdigit():
            raise GpuError(f"nvidia-smi gave {line!r} for a compute process")
        if int(pid) != own:
            others.append((int(pid), name))
    return others


def ensure_idle() -> None:
    """Refuses to go on while another process computes on the GPU."""
    listed = nvidia_smi("--query-compute-apps=pid,process_name", "--format=csv,noheader")
    others = other_processes(listed, os.getpid())
    if others:
        busy = ", ".join(f"{name} ({pid})" for pid, name in others)
        raise GpuError(f"other compute processes are using the GPU: {busy}")
