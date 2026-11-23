//! Benchmark harness: times ptxc against cuBLAS and records results as
//! JSON. The only crate that may load cuBLAS.

mod clocks;
mod cublas;
mod host;
mod problem;
mod results;
mod suites;
mod timing;

use clap::{Parser, Subcommand};
use std::process::ExitCode;
use suites::gemm::{self, GemmArgs};

/// Times ptxc against its baselines (BENCHMARKING.md).
#[derive(Debug, Parser)]
#[command(name = "ptxc-bench")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// The GEMM sweep against cuBLAS.
    Gemm(GemmArgs),
}

fn main() -> ExitCode {
    let outcome = match Args::parse().command {
        Command::Gemm(args) => match std::env::current_dir() {
            Ok(repository) => gemm::run(&args, &repository).map_err(|error| error.to_string()),
            Err(error) => Err(format!("reading the current directory: {error}")),
        },
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ptxc-bench: {error}");
            ExitCode::FAILURE
        }
    }
}
