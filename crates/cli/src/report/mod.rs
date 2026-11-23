//! Conformance reports as JSON and Markdown.

mod markdown;
mod outcome;
#[cfg(test)]
mod tests;

pub(crate) use markdown::markdown;
pub(crate) use outcome::{Failure, Outcome, Reason, Worst};

use ptxc_onnx_proto::ONNX_TAG;
use serde::Serialize;

/// One test's outcome.
#[derive(Debug, Serialize)]
pub(crate) struct TestResult {
    /// The test's name.
    pub(crate) name: String,
    /// The ONNX op it exercises, or `composite` for several.
    pub(crate) op: String,
    /// What happened.
    #[serde(flatten)]
    pub(crate) outcome: Outcome,
}

/// How many tests had each outcome.
#[derive(Debug, Default, PartialEq, Eq, Serialize)]
pub(crate) struct Totals {
    pub(crate) pass: usize,
    pub(crate) fail: usize,
    pub(crate) unsupported: usize,
}

/// A conformance run: what ran, on what, at what tolerance, and each test.
#[derive(Debug, Serialize)]
pub(crate) struct Report {
    /// The backend, as the command line names it.
    pub(crate) backend: &'static str,
    /// The ONNX release the node tests come from.
    pub(crate) onnx: &'static str,
    /// The comparison tolerance, described.
    pub(crate) tolerance: String,
    /// Counts by outcome.
    pub(crate) totals: Totals,
    /// Every test, in name order.
    pub(crate) tests: Vec<TestResult>,
}

impl Report {
    pub(crate) fn new(backend: &'static str, tolerance: String, tests: Vec<TestResult>) -> Self {
        let mut totals = Totals::default();
        for test in &tests {
            match test.outcome {
                Outcome::Pass => totals.pass += 1,
                Outcome::Fail { .. } => totals.fail += 1,
                Outcome::Unsupported { .. } => totals.unsupported += 1,
            }
        }
        Self { backend, onnx: ONNX_TAG, tolerance, totals, tests }
    }

    /// The report as pretty-printed JSON.
    pub(crate) fn json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}
