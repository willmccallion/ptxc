//! Locates downloaded data and cached reference outputs.

use ptxc_onnx_proto::ONNX_TAG;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Why the data directory or a part of it was not found.
#[derive(Debug, Error)]
pub enum DataError {
    /// No directory above the current one holds the workspace's `Cargo.toml`.
    #[error("no workspace root above {}", .from.display())]
    NoWorkspace {
        /// Where the search started.
        from: PathBuf,
    },
    /// The data has not been fetched.
    #[error("{} does not exist; run `{fetch}`", .path.display())]
    NotFetched {
        /// The missing directory.
        path: PathBuf,
        /// The command that fetches it.
        fetch: &'static str,
    },
    /// The current directory could not be read.
    #[error("reading the current directory: {0}")]
    Io(#[from] io::Error),
}

/// The workspace root: the nearest directory, from the current one up,
/// whose `Cargo.toml` declares `[workspace]`.
pub fn workspace_root() -> Result<PathBuf, DataError> {
    let from = std::env::current_dir()?;
    from.ancestors()
        .find(|dir| {
            std::fs::read_to_string(dir.join("Cargo.toml"))
                .is_ok_and(|text| is_workspace_manifest(&text))
        })
        .map(Path::to_path_buf)
        .ok_or(DataError::NoWorkspace { from })
}

fn is_workspace_manifest(text: &str) -> bool {
    text.lines().any(|line| line.trim() == "[workspace]")
}

/// The ONNX backend node tests at the pinned tag, which
/// `cargo xtask fetch onnx-tests` downloads.
pub fn onnx_node_tests() -> Result<PathBuf, DataError> {
    let path = workspace_root()?.join("data").join("onnx").join(ONNX_TAG).join("node");
    if path.is_dir() {
        Ok(path)
    } else {
        Err(DataError::NotFetched { path, fetch: "cargo xtask fetch onnx-tests" })
    }
}

#[cfg(test)]
mod tests {
    use super::is_workspace_manifest;

    #[test]
    fn only_a_workspace_section_marks_the_root() {
        assert!(is_workspace_manifest("[package]\nname = \"x\"\n\n[workspace]\nmembers = []\n"));
        assert!(!is_workspace_manifest("[package]\nname = \"x\"\n[workspace.dependencies]\n"));
        assert!(!is_workspace_manifest("[package]\nworkspace = \"..\"\n"));
    }
}
