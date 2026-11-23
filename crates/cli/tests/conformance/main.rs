//! `ptxc conformance` end to end: the real binary on node tests written to a
//! temporary directory, one per outcome.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod cases;

use cases::write_all;
use serde_json::Value;
use std::path::Path;
use std::process::Command;

/// What one run of the conformance command did.
struct Run {
    code: Option<i32>,
    stderr: String,
    report: Option<Value>,
}

/// Runs the conformance command on `tests` with stderr piped and no colour
/// settings in the environment.
fn conformance(tests: &Path, out: &Path) -> Run {
    let output = Command::new(env!("CARGO_BIN_EXE_ptxc"))
        .args(["conformance", "--backend", "interp", "--tests"])
        .arg(tests)
        .arg("--out")
        .arg(out)
        .env_remove("NO_COLOR")
        .env_remove("CLICOLOR")
        .env_remove("CLICOLOR_FORCE")
        .env_remove("RUST_LOG")
        .output()
        .unwrap();
    let report = std::fs::read_to_string(out.join("report.json"))
        .ok()
        .map(|text| serde_json::from_str(&text).unwrap());
    Run { code: output.status.code(), stderr: String::from_utf8(output.stderr).unwrap(), report }
}

fn outcome<'r>(report: &'r Value, name: &str) -> &'r Value {
    report["tests"]
        .as_array()
        .unwrap()
        .iter()
        .find(|test| test["name"] == name)
        .unwrap_or_else(|| panic!("no {name}"))
}

#[test]
fn every_outcome_is_classified_and_any_failure_fails_the_run() {
    let (tests, out) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    write_all(tests.path());

    let run = conformance(tests.path(), out.path());

    let report = run.report.unwrap();
    assert_eq!(run.code, Some(1));
    assert_eq!(report["backend"], "interp");
    assert_eq!(report["totals"], serde_json::json!({ "pass": 1, "fail": 7, "unsupported": 5 }));
    let status = |name: &str| outcome(&report, name)["status"].clone();
    let failure = |name: &str| outcome(&report, name)["failure"]["kind"].clone();
    let reason = |name: &str| outcome(&report, name)["reason"].clone();
    assert_eq!(status("test_relu"), "pass");
    assert_eq!(outcome(&report, "test_relu")["op"], "Relu");
    assert_eq!(failure("test_relu_wrong"), "mismatch");
    assert_eq!(
        outcome(&report, "test_relu_wrong")["failure"]["worst_absolute"]["position"],
        serde_json::json!([1])
    );
    assert_eq!(outcome(&report, "test_relu_second_set")["failure"]["data_set"], 1);
    assert_eq!(reason("test_topk"), serde_json::json!({ "op": "TopK" }));
    assert_eq!(reason("test_ml_domain"), serde_json::json!({ "op": "ai.onnx.ml.Binarizer" }));
    assert_eq!(reason("test_sequence"), serde_json::json!({ "dtype": "a non-tensor type" }));
    assert_eq!(reason("test_double_data"), serde_json::json!({ "dtype": "DOUBLE" }));
    assert_eq!(reason("test_cast_shape"), "value_dependent_shape");
    assert_eq!(failure("test_declared"), "import");
    assert_eq!(failure("test_gather_out_of_range"), "evaluate");
    assert_eq!(failure("test_extra_output"), "output_count");
    assert_eq!(failure("test_corrupt"), "load");
    assert_eq!(outcome(&report, "test_corrupt")["op"], "unknown");
    assert_eq!(failure("test_unreadable"), "load");
    assert!(
        std::fs::read_to_string(out.path().join("report.md"))
            .unwrap()
            .contains("| Relu | 1 | 5 | 1 |")
    );
}

#[test]
fn a_run_without_failures_succeeds() {
    let (tests, out) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    cases::relu(tests.path(), "test_relu", &[[1.0, -2.0]], &[[1.0, 0.0]]);

    let run = conformance(tests.path(), &out.path().join("nested"));

    assert_eq!(run.code, Some(0));
    assert_eq!(run.report.unwrap()["totals"]["pass"], 1);
}

#[test]
fn a_missing_tests_directory_fails_the_command() {
    let out = tempfile::tempdir().unwrap();

    let run = conformance(&out.path().join("absent"), out.path());

    assert_eq!(run.code, Some(2));
    assert!(run.report.is_none());
    assert!(run.stderr.contains("ERROR listing the node tests in"), "{}", run.stderr);
}

#[test]
fn the_terminal_shows_the_run_its_totals_and_each_failure_uncoloured_when_piped() {
    let (tests, out) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    cases::relu(tests.path(), "test_relu", &[[1.0, -2.0]], &[[1.0, 0.0]]);
    cases::relu(tests.path(), "test_relu_wrong", &[[1.0, -2.0]], &[[1.0, 5.0]]);

    let stderr = conformance(tests.path(), out.path()).stderr;

    for expected in [
        "ptxc v",
        "Conformance",
        "> Backend  interp",
        "> Tests  ",
        "passed",
        "failed",
        "unsupported",
    ] {
        assert!(stderr.contains(expected), "{expected:?} missing from\n{stderr}");
    }
    assert!(stderr.contains("ERROR test_relu_wrong: data set 0, output 0:"), "{stderr}");
    assert!(
        stderr.lines().any(|line| line.split_whitespace().collect::<Vec<_>>() == ["passed", "1"]),
        "{stderr}"
    );
    assert!(!stderr.contains('\u{1b}'), "colour escapes in piped output:\n{stderr}");
}
