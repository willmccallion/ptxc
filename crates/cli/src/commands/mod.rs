//! One module per subcommand.

mod agreement;
mod check_model;
mod compile;
mod conformance;
mod fuzz;
mod inspect;
mod run;
mod runner;
mod tune;

use crate::args::{Args, Command};
use std::process::ExitCode;

/// Runs the subcommand `args` names.
pub(crate) fn run(args: &Args) -> anyhow::Result<ExitCode> {
    match &args.command {
        Command::Compile(compile) => compile::run(compile),
        Command::Run(run) => run::run(run),
        Command::Conformance(conformance) => conformance::run(conformance),
        Command::Fuzz(fuzz) => fuzz::run(fuzz),
        Command::FuzzExport(export) => fuzz::export::run(export),
        Command::FuzzOrt(ort) => fuzz::ort::run(ort),
    }
}
