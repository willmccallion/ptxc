//! Repository tasks: regenerate protobuf code, fetch pinned test data, check crate dependency rules.

mod check_deps;
mod fetch;
mod pin;
mod proto;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage: cargo xtask proto | fetch onnx-tests";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let workspace = workspace_root();
    let outcome = match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["proto"] => proto::run(&workspace).map_err(|error| error.to_string()),
        ["fetch", "onnx-tests"] => fetch::onnx_tests(&workspace).map_err(|error| error.to_string()),
        _ => Err(USAGE.to_owned()),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("xtask: {message}");
            ExitCode::FAILURE
        }
    }
}

/// The workspace root, two levels above this crate's manifest.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
