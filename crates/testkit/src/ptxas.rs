//! Assembles PTX with ptxas, the test-only checker of emitted modules (D11).
//! The flake's dev shell and CI provide it; without it, checks fail rather
//! than skip.

use std::io::{self, Write};
use std::process::Command;
use thiserror::Error;

/// Why a module did not assemble.
#[derive(Debug, Error)]
pub enum PtxasError {
    /// ptxas could not be run.
    #[error("cannot run ptxas ({0}); it comes with the flake's dev shell (D11)")]
    Missing(io::Error),
    /// ptxas rejected the module.
    #[error("ptxas rejected the module:\n{diagnostics}\n{text}")]
    Rejected {
        /// ptxas's messages.
        diagnostics: String,
        /// The module.
        text: String,
    },
}

/// Assembles `text` for `gpu`, a target such as `sm_89`.
pub fn assemble(text: &str, gpu: &str) -> Result<(), PtxasError> {
    let mut file =
        tempfile::Builder::new().suffix(".ptx").tempfile().map_err(PtxasError::Missing)?;
    file.write_all(text.as_bytes()).map_err(PtxasError::Missing)?;
    let output = Command::new("ptxas")
        .args(["--gpu-name", gpu, "--output-file", "/dev/null"])
        .arg(file.path())
        .output()
        .map_err(PtxasError::Missing)?;
    if output.status.success() {
        return Ok(());
    }
    let diagnostics = String::from_utf8_lossy(&output.stderr).into_owned();
    Err(PtxasError::Rejected { diagnostics, text: text.to_owned() })
}
