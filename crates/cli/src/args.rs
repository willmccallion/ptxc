//! Command-line arguments: clap derive, one subcommand per command.

use clap::{Parser, Subcommand, ValueEnum};
use ptxc_core::Dim;
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
    /// Compile an ONNX model to a program file.
    Compile(CompileArgs),
    /// Run an ONNX model or program file on the GPU.
    Run(RunArgs),
    /// Run the ONNX backend node tests and write report.json and report.md.
    Conformance(ConformanceArgs),
    /// Run random graphs on every executor and shrink any mismatch.
    Fuzz(FuzzArgs),
}

/// `ptxc fuzz`'s arguments.
#[derive(Debug, clap::Args)]
pub(crate) struct FuzzArgs {
    /// The executor checked beyond the interpreters.
    #[arg(long, value_enum)]
    pub(crate) backend: Backend,
    /// The first seed.
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    /// How many seeds, from the first.
    #[arg(long, default_value_t = 1000)]
    pub(crate) cases: u64,
    /// Where shrunk mismatches are written.
    #[arg(long, default_value = "data/fuzz")]
    pub(crate) out: PathBuf,
}

/// `ptxc compile`'s arguments.
#[derive(Debug, clap::Args)]
pub(crate) struct CompileArgs {
    /// The ONNX model.
    pub(crate) model: PathBuf,
    /// The program file to write.
    #[arg(short, long)]
    pub(crate) out: PathBuf,
    /// A size for a symbolic dimension, such as `batch=8`.
    #[arg(long = "dim", value_name = "NAME=SIZE", value_parser = parse_dim)]
    pub(crate) dims: Vec<(String, Dim)>,
}

/// `ptxc run`'s arguments.
#[derive(Debug, clap::Args)]
pub(crate) struct RunArgs {
    /// A program file, or an ONNX model to compile first.
    pub(crate) model: PathBuf,
    /// The inputs in order, each a serialised ONNX `TensorProto`.
    #[arg(long = "input", value_name = "FILE.pb")]
    pub(crate) inputs: Vec<PathBuf>,
    /// The directory to write `output_0.pb`, `output_1.pb`, ... to.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    /// A size for a symbolic dimension, when compiling a model.
    #[arg(long = "dim", value_name = "NAME=SIZE", value_parser = parse_dim)]
    pub(crate) dims: Vec<(String, Dim)>,
}

/// `name=size`, as `--dim` takes it.
pub(crate) fn parse_dim(text: &str) -> Result<(String, Dim), String> {
    let (name, size) = text.split_once('=').ok_or_else(|| format!("`{text}` is not NAME=SIZE"))?;
    let size: u64 = size.parse().map_err(|_| format!("`{size}` is not a size"))?;
    if name.is_empty() {
        return Err(format!("`{text}` has no name"));
    }
    Ok((name.to_owned(), Dim::new(size)))
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
    /// Compiled and run on GPU 0.
    Gpu,
}

impl Backend {
    /// The name reports use, as the command line spells it.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Interp => "interp",
            Self::Gpu => "gpu",
        }
    }
}
