//! `ptxc conformance`: runs the ONNX backend node tests.

mod outcome;

use super::runner::Runner;
use crate::args::ConformanceArgs;
use crate::report::{Report, TestResult, markdown};
use crate::ui;
use anyhow::Context;
use ptxc_testkit::data::onnx_node_tests;
use ptxc_testkit::onnx_tests::{TestCase, TestOp, case_dirs};
use std::path::Path;
use std::process::ExitCode;

/// The exit code when a test fails.
const TEST_FAILED: u8 = 1;

/// Runs every node test on the chosen backend, writes report.json and
/// report.md, and exits with failure if any test failed.
pub(crate) fn run(args: &ConformanceArgs) -> anyhow::Result<ExitCode> {
    let root = match &args.tests {
        Some(root) => root.clone(),
        None => onnx_node_tests()?,
    };
    ui::section("Conformance");
    ui::header("Backend", args.backend.name());
    ui::header("Tests", &root.display().to_string());
    let dirs = case_dirs(&root)
        .with_context(|| format!("listing the node tests in {}", root.display()))?;
    let runner = Runner::new(args.backend)?;
    let results: Vec<TestResult> = dirs.iter().map(|dir| result(dir, &runner)).collect();
    let report = Report::new(args.backend.name(), outcome::TOLERANCE.to_string(), results);
    write_reports(args, &report)?;
    summarise(&report);
    Ok(if report.totals.fail == 0 { ExitCode::SUCCESS } else { ExitCode::from(TEST_FAILED) })
}

/// One case's result; a case that cannot be loaded fails.
fn result(dir: &Path, runner: &Runner) -> TestResult {
    let name = dir.file_name().map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    match TestCase::load(dir) {
        Ok(case) => TestResult {
            name,
            op: match case.op() {
                TestOp::Single(op) => op.clone(),
                TestOp::Composite => "composite".to_owned(),
            },
            outcome: outcome::of_case(&case, runner),
        },
        Err(error) => {
            TestResult { name, op: "unknown".to_owned(), outcome: outcome::of_load_error(&error) }
        }
    }
}

fn write_reports(args: &ConformanceArgs, report: &Report) -> anyhow::Result<()> {
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;
    for (name, text) in [("report.json", report.json()?), ("report.md", markdown(report))] {
        let path = args.out.join(name);
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

fn summarise(report: &Report) {
    let totals = &report.totals;
    ui::stat("passed", &totals.pass.to_string());
    ui::stat("failed", &totals.fail.to_string());
    ui::stat("unsupported", &totals.unsupported.to_string());
    for test in &report.tests {
        if let crate::report::Outcome::Fail { failure } = &test.outcome {
            log::error!("{}: {failure}", test.name);
        }
    }
}
