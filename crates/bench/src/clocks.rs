//! The GPU's state around a series, from `nvidia-smi`: its clocks, its
//! temperature and the reasons the driver holds its clocks down, and
//! whether another compute process is using it (BENCHMARKING.md,
//! Methodology).

use serde::{Deserialize, Serialize};
use std::process::Command;
use thiserror::Error;

/// The GPU the benchmarks run on.
const GPU: &str = "0";

/// A clock frequency in MHz.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct Megahertz(pub(crate) u32);

/// A temperature in degrees Celsius.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct Celsius(pub(crate) u32);

/// A reason the driver holds the clocks below their maximum, serialized
/// as `nvidia-smi` names it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ClockEvent {
    GpuIdle,
    SwPowerCap,
    HwSlowdown,
    SwThermalSlowdown,
    HwThermalSlowdown,
    HwPowerBrakeSlowdown,
    ApplicationsClocksSetting,
    SyncBoost,
}

/// Each reason, with the `nvidia-smi --query-gpu` field that reports it.
const EVENTS: [(ClockEvent, &str); 8] = [
    (ClockEvent::GpuIdle, "clocks_event_reasons.gpu_idle"),
    (ClockEvent::SwPowerCap, "clocks_event_reasons.sw_power_cap"),
    (ClockEvent::HwSlowdown, "clocks_event_reasons.hw_slowdown"),
    (ClockEvent::SwThermalSlowdown, "clocks_event_reasons.sw_thermal_slowdown"),
    (ClockEvent::HwThermalSlowdown, "clocks_event_reasons.hw_thermal_slowdown"),
    (ClockEvent::HwPowerBrakeSlowdown, "clocks_event_reasons.hw_power_brake_slowdown"),
    (ClockEvent::ApplicationsClocksSetting, "clocks_event_reasons.applications_clocks_setting"),
    (ClockEvent::SyncBoost, "clocks_event_reasons.sync_boost"),
];

/// The fields read before the reasons, in order.
const CLOCK_FIELDS: [&str; 3] = ["clocks.sm", "clocks.mem", "temperature.gpu"];

/// The GPU's clocks and temperature at one moment.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub(crate) struct GpuState {
    #[serde(rename = "sm_mhz")]
    pub(crate) sm: Megahertz,
    #[serde(rename = "mem_mhz")]
    pub(crate) memory: Megahertz,
    #[serde(rename = "temperature_c")]
    pub(crate) temperature: Celsius,
    /// The reasons active, in `EVENTS` order.
    pub(crate) events: Vec<ClockEvent>,
}

/// Which GPU and driver ran a suite.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct GpuIdentity {
    pub(crate) name: String,
    pub(crate) driver: String,
}

impl GpuIdentity {
    /// The benchmark GPU's name and driver version.
    pub(crate) fn read() -> Result<Self, ClockError> {
        parse_identity(&nvidia_smi(&["--query-gpu=name,driver_version", "--format=csv,noheader"])?)
    }
}

/// The identity in one line of `--query-gpu=name,driver_version` output.
fn parse_identity(line: &str) -> Result<GpuIdentity, ClockError> {
    let (name, driver) = line.trim().split_once(", ").ok_or_else(|| ClockError::Field {
        field: "name,driver_version",
        value: line.trim().to_owned(),
    })?;
    Ok(GpuIdentity { name: name.to_owned(), driver: driver.to_owned() })
}

/// A compute process using the GPU.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct ComputeProcess {
    pub(crate) pid: u32,
    pub(crate) name: String,
}

/// Why the GPU's state could not be read, or why a series cannot start.
#[derive(Debug, Error)]
pub(crate) enum ClockError {
    #[error("running nvidia-smi: {0}")]
    Run(#[from] std::io::Error),
    #[error("nvidia-smi failed: {0}")]
    Failed(String),
    #[error("nvidia-smi gave {value:?} for {field}")]
    Field { field: &'static str, value: String },
    #[error("nvidia-smi gave {0} fields where {1} were asked for")]
    FieldCount(usize, usize),
    #[error("other compute processes are using the GPU: {}", list(.0))]
    Busy(Vec<ComputeProcess>),
}

impl GpuState {
    /// The GPU's state now.
    pub(crate) fn read() -> Result<Self, ClockError> {
        let query = format!("--query-gpu={}", query_fields().join(","));
        parse_state(&nvidia_smi(&[&query, "--format=csv,noheader,nounits"])?)
    }
}

/// Fails if a compute process other than this one is using the GPU.
pub(crate) fn ensure_idle() -> Result<(), ClockError> {
    let output = nvidia_smi(&["--query-compute-apps=pid,process_name", "--format=csv,noheader"])?;
    let others = other_processes(&output, std::process::id())?;
    if others.is_empty() { Ok(()) } else { Err(ClockError::Busy(others)) }
}

/// Whether the clocks held through a series: the same before and after.
pub(crate) fn clocks_held(before: &GpuState, after: &GpuState) -> bool {
    before.sm == after.sm && before.memory == after.memory
}

fn query_fields() -> Vec<&'static str> {
    CLOCK_FIELDS.into_iter().chain(EVENTS.iter().map(|&(_, field)| field)).collect()
}

/// `nvidia-smi`'s output for the benchmark GPU, given `arguments`.
fn nvidia_smi(arguments: &[&str]) -> Result<String, ClockError> {
    let output = Command::new("nvidia-smi").args(["-i", GPU]).args(arguments).output()?;
    if !output.status.success() {
        return Err(ClockError::Failed(String::from_utf8_lossy(&output.stderr).trim().to_owned()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The state in one line of `--format=csv,noheader,nounits` output for
/// `query_fields`.
fn parse_state(line: &str) -> Result<GpuState, ClockError> {
    let fields = query_fields();
    let values: Vec<&str> = line.trim().split(", ").collect();
    if values.len() != fields.len() {
        return Err(ClockError::FieldCount(values.len(), fields.len()));
    }
    let number = |index: usize| -> Result<u32, ClockError> {
        values[index].parse().map_err(|_| ClockError::Field {
            field: fields[index],
            value: values[index].to_owned(),
        })
    };
    let mut events = Vec::new();
    for (offset, &(event, field)) in EVENTS.iter().enumerate() {
        match values[CLOCK_FIELDS.len() + offset] {
            "Active" => events.push(event),
            "Not Active" => {}
            other => return Err(ClockError::Field { field, value: other.to_owned() }),
        }
    }
    Ok(GpuState {
        sm: Megahertz(number(0)?),
        memory: Megahertz(number(1)?),
        temperature: Celsius(number(2)?),
        events,
    })
}

/// The processes in `--query-compute-apps=pid,process_name` output other
/// than `own`.
fn other_processes(output: &str, own: u32) -> Result<Vec<ComputeProcess>, ClockError> {
    let mut others = Vec::new();
    for line in output.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let (pid, name) = line.split_once(", ").ok_or_else(|| ClockError::Field {
            field: "compute process",
            value: line.to_owned(),
        })?;
        let pid =
            pid.parse().map_err(|_| ClockError::Field { field: "pid", value: pid.to_owned() })?;
        if pid != own {
            others.push(ComputeProcess { pid, name: name.to_owned() });
        }
    }
    Ok(others)
}

fn list(processes: &[ComputeProcess]) -> String {
    let each: Vec<String> =
        processes.iter().map(|process| format!("{} ({})", process.name, process.pid)).collect();
    each.join(", ")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{
        Celsius, ClockError, ClockEvent, ComputeProcess, GpuIdentity, GpuState, Megahertz,
        clocks_held, other_processes, parse_identity, parse_state,
    };

    /// `nvidia-smi`'s answer on an idle RTX 4060 Ti.
    const IDLE: &str = "210, 405, 45, Active, Not Active, Not Active, Not Active, Not Active, Not Active, Not Active, Not Active\n";

    fn state(sm: u32, memory: u32) -> GpuState {
        GpuState {
            sm: Megahertz(sm),
            memory: Megahertz(memory),
            temperature: Celsius(60),
            events: vec![],
        }
    }

    #[test]
    fn a_state_line_gives_the_clocks_temperature_and_active_reasons() {
        let read = parse_state(IDLE).unwrap();

        let expected = GpuState {
            events: vec![ClockEvent::GpuIdle],
            temperature: Celsius(45),
            ..state(210, 405)
        };
        assert_eq!(read, expected);
    }

    #[test]
    fn every_reason_is_read_in_its_place() {
        let line = "2520, 9001, 71, Not Active, Active, Not Active, Not Active, Active, Not Active, Not Active, Active";

        let read = parse_state(line).unwrap();

        let expected =
            vec![ClockEvent::SwPowerCap, ClockEvent::HwThermalSlowdown, ClockEvent::SyncBoost];
        assert_eq!(read.events, expected);
    }

    #[test]
    fn an_unreadable_field_is_an_error_naming_it() {
        let unsupported = IDLE.replacen("Not Active", "[N/A]", 1);

        let error = parse_state(&unsupported).unwrap_err();
        let short = parse_state("210, 405").unwrap_err();
        let garbled = parse_state(&IDLE.replacen("405", "fast", 1)).unwrap_err();

        assert!(
            matches!(error, ClockError::Field { field: "clocks_event_reasons.sw_power_cap", .. }),
            "{error}"
        );
        assert!(matches!(short, ClockError::FieldCount(2, 11)), "{short}");
        assert!(matches!(garbled, ClockError::Field { field: "clocks.mem", .. }), "{garbled}");
    }

    #[test]
    fn the_identity_is_the_gpus_name_and_driver() {
        let identity = parse_identity("NVIDIA GeForce RTX 4060 Ti, 595.112.07\n").unwrap();

        let expected = GpuIdentity {
            name: "NVIDIA GeForce RTX 4060 Ti".to_owned(),
            driver: "595.112.07".to_owned(),
        };
        assert_eq!(identity, expected);
        assert!(parse_identity("no comma").is_err());
    }

    #[test]
    fn clocks_hold_only_if_both_are_unchanged() {
        assert!(clocks_held(&state(2520, 9001), &state(2520, 9001)));
        assert!(!clocks_held(&state(2520, 9001), &state(2505, 9001)));
        assert!(!clocks_held(&state(2520, 9001), &state(2520, 8001)));
    }

    #[test]
    fn this_process_is_not_another_compute_process() {
        let output = "4242, ptxc-bench\n77, /usr/bin/python3\n";

        let others = other_processes(output, 4242).unwrap();

        assert_eq!(others, [ComputeProcess { pid: 77, name: "/usr/bin/python3".to_owned() }]);
        assert_eq!(other_processes("", 4242).unwrap(), []);
    }

    #[test]
    fn a_busy_gpu_is_refused_naming_the_processes() {
        let busy = ClockError::Busy(other_processes("77, python3\n", 1).unwrap());

        assert_eq!(busy.to_string(), "other compute processes are using the GPU: python3 (77)");
    }
}

#[cfg(all(test, feature = "gpu"))]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod gpu {
    use super::{ClockError, GpuIdentity, GpuState, ensure_idle};
    use ptxc_cuda::{DeviceOrdinal, Driver};
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};

    /// What the child prints once its context exists.
    const READY: &str = "context ready";

    #[test]
    #[ignore = "the child process of a_gpu_another_process_uses_is_refused"]
    fn hold_a_context_until_stdin_closes() {
        let _device = Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap();
        println!("{READY}");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        let _read = std::io::stdin().read_line(&mut line).unwrap();
    }

    #[test]
    fn the_gpu_reports_its_clocks_and_a_gpu_another_process_uses_is_refused() {
        let _gpu = crate::timing::exclusive_gpu();
        let state = GpuState::read().unwrap();
        let identity = GpuIdentity::read().unwrap();
        ensure_idle().unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "clocks::gpu::hold_a_context_until_stdin_closes"])
            .args(["--ignored", "--nocapture"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
        while lines.next().unwrap().unwrap().trim() != READY {}

        let busy = ensure_idle();

        drop(child.stdin.take());
        let _status = child.wait().unwrap();
        assert!(state.sm.0 > 0 && state.memory.0 > 0, "{state:?}");
        assert!(identity.name.starts_with("NVIDIA"), "{identity:?}");
        let Err(ClockError::Busy(others)) = busy else { panic!("{busy:?}") };
        assert!(others.iter().any(|process| process.pid == child.id()), "{others:?}");
    }
}
