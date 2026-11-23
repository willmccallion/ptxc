//! The reports of a small run: report.md against a golden file, and the
//! JSON form of each outcome.

use super::{Failure, Outcome, Reason, Report, TestResult, Worst, markdown};

fn result(name: &str, op: &str, outcome: Outcome) -> TestResult {
    TestResult { name: name.to_owned(), op: op.to_owned(), outcome }
}

fn sample() -> Report {
    let mismatch = Failure::Mismatch {
        data_set: 0,
        output: 1,
        message: "1 of 4 elements differ".to_owned(),
        worst_absolute: Some(Worst {
            position: vec![0, 2],
            actual: 1.5,
            expected: 1.0,
            error: 0.5,
        }),
        worst_relative: Some(Worst {
            position: vec![0, 2],
            actual: 1.5,
            expected: f64::NAN,
            error: f64::INFINITY,
        }),
    };
    let tests = vec![
        result("test_add", "Add", Outcome::Pass),
        result("test_add_bcast", "Add", Outcome::Fail { failure: mismatch }),
        result(
            "test_add_double",
            "Add",
            Outcome::Unsupported { reason: Reason::DType("DOUBLE".to_owned()) },
        ),
        result("test_topk", "TopK", Outcome::Unsupported { reason: Reason::Op("TopK".to_owned()) }),
        result(
            "test_topk_neg",
            "TopK",
            Outcome::Unsupported { reason: Reason::Op("TopK".to_owned()) },
        ),
        result(
            "test_layer_norm_expanded",
            "composite",
            Outcome::Fail { failure: Failure::Evaluate { message: "boom".to_owned() } },
        ),
        result(
            "test_loop",
            "Loop",
            Outcome::Unsupported { reason: Reason::ControlFlow("Loop".to_owned()) },
        ),
    ];
    Report::new("interp", "atol 1e-7, rtol 1e-3".to_owned(), tests)
}

#[test]
fn the_markdown_report_totals_by_status_op_failure_and_reason() {
    ptxc_testkit::golden::check("report.md", &markdown(&sample())).unwrap();
}

#[test]
fn the_json_report_tags_each_outcome_and_writes_non_finite_values_as_text() {
    let json: serde_json::Value = serde_json::from_str(&sample().json().unwrap()).unwrap();

    assert_eq!(json["backend"], "interp");
    assert_eq!(json["onnx"], ptxc_onnx_proto::ONNX_TAG);
    assert_eq!(json["totals"], serde_json::json!({ "pass": 1, "fail": 2, "unsupported": 4 }));
    assert_eq!(
        json["tests"][0],
        serde_json::json!({ "name": "test_add", "op": "Add", "status": "pass" })
    );
    let failure = &json["tests"][1];
    assert_eq!(failure["status"], "fail");
    assert_eq!(failure["failure"]["kind"], "mismatch");
    assert_eq!(
        failure["failure"]["worst_relative"],
        serde_json::json!({ "position": [0, 2], "actual": "1.5", "expected": "NaN", "error": "inf" })
    );
    assert_eq!(json["tests"][2]["reason"], serde_json::json!({ "dtype": "DOUBLE" }));
    assert_eq!(json["tests"][6]["reason"], serde_json::json!({ "control_flow": "Loop" }));
}

#[test]
fn reasons_and_failures_read_as_sentences() {
    let attribute = Reason::Attribute { op: "Dropout".to_owned(), attribute: "mask".to_owned() };

    assert_eq!(attribute.to_string(), "Dropout attribute mask");
    assert_eq!(Reason::Opset(7).to_string(), "opset 7");
    assert_eq!(Reason::DynamicShape("N".to_owned()).to_string(), "dynamic shape (N)");
    assert_eq!(Reason::ValueDependentShape.to_string(), "value-dependent shape");
    let count = Failure::OutputCount { data_set: 1, expected: 2, actual: 1 };
    assert_eq!(count.to_string(), "data set 1: 1 outputs where 2 were expected");
    assert_eq!(Failure::Load { message: "x".to_owned() }.to_string(), "loading: x");
    assert_eq!(Failure::Import { message: "x".to_owned() }.to_string(), "importing: x");
    assert_eq!(
        Failure::LoopInterpreter { data_set: 2, message: "x".to_owned() }.to_string(),
        "data set 2: the GPU differs from the loop interpreter: x"
    );
}
