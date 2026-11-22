//! Repository tasks: regenerate protobuf code, fetch pinned test data, check crate dependency rules.

mod check_deps;
mod fetch;
mod proto;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage: cargo xtask proto";

fn main() -> ExitCode {
    let task = std::env::args().nth(1);
    let outcome = match task.as_deref() {
        Some("proto") => proto::run(&workspace_root()).map_err(|error| error.to_string()),
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
