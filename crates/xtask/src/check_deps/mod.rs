//! `cargo xtask check-deps`: enforces the dependency rules of
//! ARCHITECTURE.md from `cargo metadata` and the crates' sources.

mod crates;
mod rules;
mod sources;
#[cfg(test)]
mod tests;

use crates::Workspace;
use rules::Violation;
use std::path::Path;
use std::process::Command;
use thiserror::Error;

/// Why the rules could not be checked, or which they found broken.
#[derive(Debug, Error)]
pub(crate) enum CheckDepsError {
    #[error("running cargo metadata: {0}")]
    Metadata(String),
    #[error("parsing cargo metadata's output: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("reading the sources: {0}")]
    Read(#[from] std::io::Error),
    #[error("{} dependency rules broken:\n{}", .0.len(), list(.0))]
    Broken(Vec<Violation>),
}

/// Checks every rule, failing with each one broken.
pub(crate) fn run(root: &Path) -> Result<(), CheckDepsError> {
    let root = std::fs::canonicalize(root)?;
    let root = root.as_path();
    let workspace = Workspace::parse(&metadata(root)?)?;
    let manifests = sources::manifests(root, &workspace)?;
    let files = sources::sources(root, &workspace)?;
    let broken = rules::violations(&workspace, &manifests, &files);
    if broken.is_empty() { Ok(()) } else { Err(CheckDepsError::Broken(broken)) }
}

/// `cargo metadata`'s description of the workspace, without its
/// dependencies' packages, so it needs no network.
fn metadata(root: &Path) -> Result<String, CheckDepsError> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(root)
        .output()
        .map_err(|error| CheckDepsError::Metadata(error.to_string()))?;
    if !output.status.success() {
        return Err(CheckDepsError::Metadata(String::from_utf8_lossy(&output.stderr).into()));
    }
    String::from_utf8(output.stdout).map_err(|error| CheckDepsError::Metadata(error.to_string()))
}

fn list(violations: &[Violation]) -> String {
    violations.iter().map(|violation| format!("  {violation}")).collect::<Vec<_>>().join("\n")
}
