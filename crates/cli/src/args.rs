//! Command-line arguments: clap derive, one subcommand per command.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// The ptxc command.
#[derive(Debug, Parser)]
#[command(author, version, about = "ptxc: a deep-learning compiler from ONNX to PTX", long_about = None)]
pub(crate) struct Args {
    /// What to do.
    #[command(subcommand)]
    pub(crate) command: Command,
}

/// One subcommand.
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Run the ONNX backend node tests and write report.json and report.md.
    Conformance(ConformanceArgs),
}

/// `ptxc conformance`'s arguments.
#[derive(Debug, clap::Args)]
pub(crate) struct ConformanceArgs {
    /// The executor that runs the tests.
    #[arg(long, value_enum)]
    pub(crate) backend: Backend,
    /// The node tests directory; by default the ones `cargo xtask fetch
    /// onnx-tests` downloads.
    #[arg(long)]
    pub(crate) tests: Option<PathBuf>,
    /// The directory report.json and report.md are written to.
    #[arg(long, default_value = ".")]
    pub(crate) out: PathBuf,
}

/// An executor that runs graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum Backend {
    /// The reference interpreter.
    Interp,
}

impl Backend {
    /// The name reports use, as the command line spells it.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Interp => "interp",
        }
    }
}
