//! `report.md`: totals by outcome and by op, every failure, and the
//! unsupported tests by reason.

use super::{Outcome, Reason, Report};
use std::collections::BTreeMap;

/// Counts of one op's tests by outcome.
#[derive(Default)]
struct OpCounts {
    pass: usize,
    fail: usize,
    unsupported: usize,
}

/// The report as Markdown.
pub(crate) fn markdown(report: &Report) -> String {
    let totals = &report.totals;
    let mut lines = vec![
        format!("# Conformance: {} on the ONNX {} node tests", report.backend, report.onnx),
        String::new(),
        format!("Compared at {}.", report.tolerance),
        String::new(),
        "| Status | Tests |".to_owned(),
        "|---|---:|".to_owned(),
        format!("| pass | {} |", totals.pass),
        format!("| fail | {} |", totals.fail),
        format!("| unsupported | {} |", totals.unsupported),
    ];
    lines.extend(by_op(report));
    lines.extend(failures(report));
    lines.extend(reasons(report));
    lines.push(String::new());
    lines.join("\n")
}

fn by_op(report: &Report) -> Vec<String> {
    let mut counts: BTreeMap<&str, OpCounts> = BTreeMap::new();
    for test in &report.tests {
        let entry = counts.entry(test.op.as_str()).or_default();
        match test.outcome {
            Outcome::Pass => entry.pass += 1,
            Outcome::Fail { .. } => entry.fail += 1,
            Outcome::Unsupported { .. } => entry.unsupported += 1,
        }
    }
    let header = ["", "## By op", "", "| Op | Pass | Fail | Unsupported |", "|---|---:|---:|---:|"];
    let rows = counts
        .into_iter()
        .map(|(op, c)| format!("| {op} | {} | {} | {} |", c.pass, c.fail, c.unsupported));
    header.into_iter().map(str::to_owned).chain(rows).collect()
}

fn failures(report: &Report) -> Vec<String> {
    let rows: Vec<String> = report
        .tests
        .iter()
        .filter_map(|test| match &test.outcome {
            Outcome::Fail { failure } => Some(format!("- `{}`: {failure}", test.name)),
            _ => None,
        })
        .collect();
    if rows.is_empty() {
        return rows;
    }
    ["", "## Failures", ""].into_iter().map(str::to_owned).chain(rows).collect()
}

fn reasons(report: &Report) -> Vec<String> {
    let mut counts: BTreeMap<&Reason, usize> = BTreeMap::new();
    for test in &report.tests {
        if let Outcome::Unsupported { reason } = &test.outcome {
            *counts.entry(reason).or_default() += 1;
        }
    }
    if counts.is_empty() {
        return Vec::new();
    }
    let mut ordered: Vec<_> = counts.into_iter().collect();
    ordered.sort_by(|(a, x), (b, y)| y.cmp(x).then_with(|| a.cmp(b)));
    let header = ["", "## Unsupported, by reason", "", "| Reason | Tests |", "|---|---:|"];
    let rows = ordered.into_iter().map(|(reason, count)| format!("| {reason} | {count} |"));
    header.into_iter().map(str::to_owned).chain(rows).collect()
}
