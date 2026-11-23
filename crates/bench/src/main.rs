//! Benchmark harness: times ptxc against cuBLAS and records results as JSON. The only crate that may load cuBLAS.

#[cfg_attr(
    not(all(test, feature = "gpu")),
    expect(dead_code, reason = "the GEMM sweep (#97) is its first caller")
)]
mod clocks;
#[cfg_attr(
    not(all(test, feature = "gpu")),
    expect(dead_code, reason = "the GEMM sweep (#97) is its first caller")
)]
mod cublas;
#[cfg_attr(
    not(all(test, feature = "gpu")),
    expect(dead_code, reason = "the GEMM sweep (#97) is its first caller")
)]
mod problem;
#[expect(dead_code, reason = "the GEMM sweep (#97) is its first caller")]
mod results;
mod suites;
#[cfg_attr(
    not(all(test, feature = "gpu")),
    expect(dead_code, reason = "the GEMM sweep (#97) is its first caller")
)]
mod timing;

use std::process::ExitCode;

fn main() -> ExitCode {
    eprintln!("ptxc-bench: nothing is implemented yet; see docs/ROADMAP.md");
    ExitCode::FAILURE
}
