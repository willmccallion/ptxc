//! `ptxc fuzz-export` and `ptxc fuzz-ort` end to end, through the real
//! binary, without ONNX Runtime: the exported cases themselves, and a
//! check with no cached outputs to compare.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

use ptxc_core::DType;
use ptxc_graph::{Op, ReduceOp};
use ptxc_onnx::{ImportOptions, import_model};
use ptxc_testkit::reference::inputs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Cases exported from seed 0: enough that the first seeds ONNX Runtime
/// cannot run, with a bf16 input or a mean over no elements, are passed.
const EXPORTED: usize = 60;

fn ptxc(arguments: &[&str], path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ptxc")).args(arguments).arg(path).output().unwrap()
}

fn case_directories(root: &Path) -> Vec<PathBuf> {
    let mut cases: Vec<PathBuf> =
        std::fs::read_dir(root).unwrap().map(|entry| entry.unwrap().path()).collect();
    cases.sort();
    cases
}

/// Whether ONNX Runtime can run the case in `directory`: it has no bf16
/// kernels, and gives 0 for a mean over no elements, which ONNX leaves
/// undefined.
fn onnx_runtime_runs(directory: &Path) -> bool {
    let bytes = std::fs::read(directory.join("model.onnx")).unwrap();
    let graph = import_model(&bytes, &ImportOptions::default()).unwrap();
    let tensors = inputs(&directory.join("inputs")).unwrap();
    let empty_mean = graph.nodes().any(|(_, node)| {
        let (Op::Reduce(attrs), Some(&input)) = (node.op(), node.operands().iter().next()) else {
            return false;
        };
        let dims = graph.value(input).ty().shape().dims();
        attrs.op == ReduceOp::Mean && attrs.axes.iter().any(|axis| dims[axis.index()].get() == 0)
    });
    tensors.iter().all(|tensor| tensor.dtype() != DType::Bf16) && !empty_mean
}

#[test]
fn fuzz_export_writes_the_cases_asked_for_and_only_ones_onnx_runtime_runs() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("cases");
    let count = EXPORTED.to_string();

    let output = ptxc(&["fuzz-export", "--seed", "0", "--cases", &count, "--out"], &out);

    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let cases = case_directories(&out);
    assert_eq!(cases.len(), EXPORTED);
    for case in cases {
        assert!(onnx_runtime_runs(&case), "{}", case.display());
    }
}

#[test]
fn fuzz_ort_with_no_cached_outputs_to_compare_fails() {
    let dir = tempfile::tempdir().unwrap();

    let output = ptxc(&["fuzz-ort", "--backend", "interp", "--dir"], dir.path());

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(stderr.contains("has ONNX Runtime outputs"), "{stderr}");
}
