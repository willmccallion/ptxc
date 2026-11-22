//! The ptxc command: compile, run, tune, conformance, fuzz, check-model and inspect.

mod args;
mod commands;
mod logger;
mod phase;
mod report;
mod ui;

use std::process::ExitCode;

fn main() -> ExitCode {
    eprintln!("ptxc: nothing is implemented yet; see docs/ROADMAP.md");
    ExitCode::FAILURE
}
