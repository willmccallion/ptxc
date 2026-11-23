//! The ptxc command: compile, run, tune, conformance, fuzz, check-model and inspect.

mod args;
mod commands;
mod logger;
mod phase;
mod report;
mod ui;

use clap::Parser;
use std::process::ExitCode;

/// The exit code when the command itself fails, as opposed to a test.
const COMMAND_FAILED: u8 = 2;

fn main() -> ExitCode {
    logger::init();
    let args = args::Args::parse();
    ui::banner();
    commands::run(&args).unwrap_or_else(|error| {
        log::error!("{error:#}");
        ExitCode::from(COMMAND_FAILED)
    })
}
