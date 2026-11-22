//! Repository tasks: regenerate protobuf code, fetch pinned test data, check crate dependency rules.

mod check_deps;
mod fetch;
mod proto;

use std::process::ExitCode;

fn main() -> ExitCode {
    eprintln!("xtask: nothing is implemented yet; see docs/ROADMAP.md");
    ExitCode::FAILURE
}
