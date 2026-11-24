//! Assembles PTX with ptxas, the test-only checker of emitted modules (D11).
//! The flake's dev shell and CI provide it; without it, checks fail rather
//! than skip.

use ptxc_core::ByteSize;
use std::io::{self, Write};
use std::process::Command;
use thiserror::Error;

/// Why a module did not assemble.
#[derive(Debug, Error)]
pub enum PtxasError {
    /// ptxas could not be run.
    #[error("cannot run ptxas ({0}); it comes with the flake's dev shell (D11)")]
    Missing(io::Error),
    /// ptxas's report lacks what was asked of it.
    #[error("ptxas's report says nothing of a stack frame:\n{0}")]
    Report(String),
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
    run(text, gpu, &[]).map(drop)
}

/// Assembles `text`, a module of one function, for `gpu`, and returns the
/// local memory its stack frame takes per thread, as ptxas reports it.
pub fn stack_frame(text: &str, gpu: &str) -> Result<ByteSize, PtxasError> {
    let report = run(text, gpu, &["--verbose"])?;
    let frame = report.lines().find_map(|line| line.split_once(" bytes stack frame"));
    frame
        .and_then(|(bytes, _)| bytes.trim().parse().ok())
        .map(ByteSize::new)
        .ok_or_else(|| PtxasError::Report(report.clone()))
}

/// ptxas's report on assembling `text` for `gpu` with `options`.
fn run(text: &str, gpu: &str, options: &[&str]) -> Result<String, PtxasError> {
    let mut file =
        tempfile::Builder::new().suffix(".ptx").tempfile().map_err(PtxasError::Missing)?;
    file.write_all(text.as_bytes()).map_err(PtxasError::Missing)?;
    let output = Command::new("ptxas")
        .args(["--gpu-name", gpu, "--output-file", "/dev/null"])
        .args(options)
        .arg(file.path())
        .output()
        .map_err(PtxasError::Missing)?;
    let report = String::from_utf8_lossy(&output.stdout) + String::from_utf8_lossy(&output.stderr);
    if output.status.success() {
        return Ok(report.into_owned());
    }
    Err(PtxasError::Rejected { diagnostics: report.into_owned(), text: text.to_owned() })
}

#[cfg(test)]
mod tests {
    use super::{PtxasError, assemble};

    #[test]
    fn a_malformed_module_is_rejected_with_ptxas_diagnostics() {
        let rejected = assemble(".version 8.4\n.target sm_89\nnot an instruction;\n", "sm_89");

        let Err(PtxasError::Rejected { diagnostics, .. }) = rejected else {
            panic!("ptxas accepted a malformed module: {rejected:?}")
        };
        assert!(diagnostics.contains("error"), "{diagnostics}");
    }
}
