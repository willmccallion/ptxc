//! Benchmark harness: times ptxc against cuBLAS and records results as JSON. The only crate that may load cuBLAS.

mod clocks;
mod cublas;
mod results;
mod suites;
mod timing;

use std::process::ExitCode;

fn main() -> ExitCode {
    eprintln!("ptxc-bench: nothing is implemented yet; see docs/ROADMAP.md");
    ExitCode::FAILURE
}
