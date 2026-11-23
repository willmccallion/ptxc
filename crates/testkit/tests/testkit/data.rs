//! Finding the workspace and its data.

use ptxc_testkit::data::{DataError, onnx_node_tests, workspace_root};
use std::path::Path;

#[test]
fn the_workspace_root_is_found_from_a_member_crate() {
    let root = workspace_root().unwrap();

    assert!(root.join("crates").join("testkit").join("Cargo.toml").is_file(), "{}", root.display());
}

#[test]
fn the_node_tests_live_under_the_pinned_tag() {
    let expected =
        workspace_root().unwrap().join("data/onnx").join(ptxc_onnx_proto::ONNX_TAG).join("node");

    let path = match onnx_node_tests() {
        Ok(path) => path,
        Err(DataError::NotFetched { path, fetch }) => {
            assert_eq!(fetch, "cargo xtask fetch onnx-tests");
            path
        }
        Err(other) => panic!("{other}"),
    };

    assert_eq!(path, expected);
    assert!(Path::new(ptxc_onnx_proto::ONNX_TAG).to_string_lossy().starts_with('v'));
}
