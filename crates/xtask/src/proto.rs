//! `cargo xtask proto`: regenerates the ONNX protobuf types from the
//! committed `onnx.proto` (D7, D30).

use crate::pin::ONNX_TAG;
use prost_build::Module;
use std::path::Path;
use thiserror::Error;

const PROTO_DIR: &str = "crates/onnx-proto/proto";
const PROTO_FILE: &str = "onnx.proto";
const GENERATED_FILE: &str = "crates/onnx-proto/src/generated.rs";

/// A failure to regenerate the ONNX protobuf types.
#[derive(Debug, Error)]
pub(crate) enum ProtoError {
    #[error("parsing {PROTO_DIR}/{PROTO_FILE}: {0}")]
    Parse(#[from] protox::Error),
    #[error("generating Rust from {PROTO_FILE}: {0}")]
    Generate(std::io::Error),
    #[error("expected {PROTO_FILE} to generate one module, got {0}")]
    ModuleCount(usize),
    #[error("writing {GENERATED_FILE}: {0}")]
    Write(std::io::Error),
}

/// Writes the Rust types for `onnx.proto` to `generated.rs`.
pub(crate) fn run(workspace: &Path) -> Result<(), ProtoError> {
    let code = generate(workspace)?;
    std::fs::write(workspace.join(GENERATED_FILE), code).map_err(ProtoError::Write)
}

/// The Rust source for `onnx.proto`, as `generated.rs` should hold it.
fn generate(workspace: &Path) -> Result<String, ProtoError> {
    let descriptors = protox::compile([PROTO_FILE], [workspace.join(PROTO_DIR)])?;
    let requests = descriptors
        .file
        .into_iter()
        .map(|file| (Module::from_protobuf_package_name(file.package()), file))
        .collect();
    let modules = prost_build::Config::new().generate(requests).map_err(ProtoError::Generate)?;
    let [code]: [String; 1] = modules
        .into_values()
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|codes: Vec<String>| ProtoError::ModuleCount(codes.len()))?;
    Ok(format!(
        "// From onnx.proto at ONNX {ONNX_TAG}. Regenerate with `cargo xtask proto`.\n\
         /// The ONNX release `onnx.proto` and the node tests come from (D31).\n\
         pub const ONNX_TAG: &str = \"{ONNX_TAG}\";\n\n{code}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_generated_code_is_what_the_generator_produces() {
        let workspace = crate::workspace_root();
        let committed = std::fs::read_to_string(workspace.join(GENERATED_FILE)).unwrap();

        let regenerated = generate(&workspace).unwrap();

        assert!(regenerated == committed, "{GENERATED_FILE} is stale; run `cargo xtask proto`");
    }
}
