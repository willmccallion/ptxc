//! The manifests and Rust sources the rules read.

use super::crates::Workspace;
use std::io;
use std::path::{Path, PathBuf};

/// This check's own sources, which name what they look for.
// NOTE: rules 3 and 4 forbid any mention of cuBLAS or `unsafe_code`
// outside `bench` and `cuda`; the check that enforces them is the one
// place that must spell both out.
const THIS_CHECK: &str = "crates/xtask/src/check_deps";

/// The root manifest and each crate's.
#[derive(Debug)]
pub(super) struct Manifests {
    /// The workspace's `Cargo.toml`.
    pub(super) root: String,
    /// Each crate's name and `Cargo.toml`.
    pub(super) crates: Vec<(String, String)>,
}

/// A Rust source or manifest in a crate.
#[derive(Debug)]
pub(super) struct SourceFile {
    /// The crate it belongs to.
    pub(super) krate: String,
    /// Its path from the workspace root.
    pub(super) path: String,
    /// Its text.
    pub(super) text: String,
}

/// Every manifest of `workspace`, rooted at `root`.
pub(super) fn manifests(root: &Path, workspace: &Workspace) -> io::Result<Manifests> {
    let crates = workspace
        .crates
        .iter()
        .map(|krate| Ok((krate.name.clone(), read(&krate.manifest)?)))
        .collect::<io::Result<_>>()?;
    Ok(Manifests { root: read(&root.join("Cargo.toml"))?, crates })
}

/// Every `.rs` file and manifest under each crate's directory, except this
/// check's own.
pub(super) fn sources(root: &Path, workspace: &Workspace) -> io::Result<Vec<SourceFile>> {
    let skipped = root.join(THIS_CHECK);
    let mut files = Vec::new();
    for krate in &workspace.crates {
        let Some(directory) = krate.manifest.parent() else { continue };
        for path in files_under(directory, &skipped)? {
            let shown = path.strip_prefix(root).unwrap_or(&path).display().to_string();
            files.push(SourceFile { krate: krate.name.clone(), path: shown, text: read(&path)? });
        }
    }
    Ok(files)
}

/// The `.rs` files and `Cargo.toml`s under `directory`, outside `skipped`.
pub(super) fn files_under(directory: &Path, skipped: &Path) -> io::Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next)? {
            let path = entry?.path();
            if path.starts_with(skipped) {
                continue;
            }
            if path.is_dir() {
                pending.push(path);
            } else if is_source(&path) {
                found.push(path);
            }
        }
    }
    Ok(found)
}

fn is_source(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "rs")
        || path.file_name().is_some_and(|name| name == "Cargo.toml")
}

fn read(path: &Path) -> io::Result<String> {
    std::fs::read_to_string(path)
        .map_err(|error| io::Error::new(error.kind(), format!("{}: {error}", path.display())))
}
