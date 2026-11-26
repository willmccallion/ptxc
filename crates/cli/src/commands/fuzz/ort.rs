//! `ptxc fuzz-ort`: checks each exported fuzz case's loop-interpreter
//! outputs against ONNX Runtime's cached ones, within tolerance, and the
//! backend against the loop interpreter exactly (D54).

use super::check::{Backend, Expected, Failure, check_graph};
use super::export::{INPUTS, MODEL};
use super::{MISMATCHED, selected};
use crate::args::FuzzOrtArgs;
use crate::commands::runner::Runner;
use crate::ui;
use anyhow::Context;
use ptxc_graph::Graph;
use ptxc_onnx::{ImportOptions, import_model};
use ptxc_testkit::reference::{ReferenceError, inputs, ort_reference_under, reference_root};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// What the cases came to.
#[derive(Debug, Default)]
struct Tally {
    agreed: u64,
    rejected: u64,
    /// Cases without ONNX Runtime outputs: not yet run, or ONNX Runtime has
    /// no kernel for one of their ops and types.
    unreferenced: u64,
    mismatched: u64,
}

impl Tally {
    /// Counts what a case with ONNX Runtime outputs came to.
    const fn record(&mut self, outcome: &Result<(), Failure>) {
        match outcome {
            Ok(()) => self.agreed += 1,
            Err(Failure::Rejected(_)) => self.rejected += 1,
            Err(Failure::Mismatch(_)) => self.mismatched += 1,
        }
    }
}

/// Checks every case under `args.dir`, failing if one mismatched or none
/// had ONNX Runtime outputs to compare.
pub(crate) fn run(args: &FuzzOrtArgs) -> anyhow::Result<ExitCode> {
    ui::section("Fuzz against ONNX Runtime");
    ui::header("Backend", args.backend.name());
    ui::header("Cases", &args.dir.display().to_string());
    let runner = Runner::new(args.backend)?;
    let on_runner = |graph: &Graph, inputs: &[_]| runner.evaluate(graph, inputs);
    let backend = selected(args.backend, &on_runner);
    let tally = check_cases(&args.dir, &reference_root()?, backend)?;
    ui::stat("agreed", &tally.agreed.to_string());
    ui::stat("rejected by the loop interpreter", &tally.rejected.to_string());
    ui::stat("without ONNX Runtime outputs", &tally.unreferenced.to_string());
    ui::stat("mismatched", &tally.mismatched.to_string());
    verdict(&tally, &args.dir)
}

/// Every case under `directory` checked against the outputs cached under
/// `references`.
fn check_cases(
    directory: &Path,
    references: &Path,
    backend: Option<&Backend<'_>>,
) -> anyhow::Result<Tally> {
    let mut tally = Tally::default();
    for case in case_directories(directory)? {
        check_case(&case, references, backend, &mut tally)?;
    }
    Ok(tally)
}

/// Success unless a case under `directory` mismatched; an error if none had
/// ONNX Runtime outputs to compare.
fn verdict(tally: &Tally, directory: &Path) -> anyhow::Result<ExitCode> {
    if tally.agreed + tally.mismatched == 0 {
        anyhow::bail!(
            "no case under {} has ONNX Runtime outputs; run \
             `uv run --project baselines ptxc-baselines fuzz-reference --dir {}` first",
            directory.display(),
            directory.display()
        );
    }
    let passed = tally.mismatched == 0;
    Ok(if passed { ExitCode::SUCCESS } else { ExitCode::from(MISMATCHED) })
}

/// The case directories under `directory`, in name order.
fn case_directories(directory: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let entries =
        std::fs::read_dir(directory).with_context(|| format!("reading {}", directory.display()))?;
    let mut cases = Vec::new();
    for entry in entries {
        let path = entry.with_context(|| format!("reading {}", directory.display()))?.path();
        if path.is_dir() {
            cases.push(path);
        }
    }
    cases.sort();
    Ok(cases)
}

/// Checks the case in `directory` against the outputs cached under
/// `references`, counting what it came to.
fn check_case(
    directory: &Path,
    references: &Path,
    backend: Option<&Backend<'_>>,
    tally: &mut Tally,
) -> anyhow::Result<()> {
    let (model, input_set) = (directory.join(MODEL), directory.join(INPUTS));
    let expected = match ort_reference_under(references, &model, &input_set) {
        Ok(outputs) => outputs,
        Err(ReferenceError::Missing(_)) => {
            tally.unreferenced += 1;
            return Ok(());
        }
        Err(error) => {
            return Err(error).context(format!("reading {}'s reference", directory.display()));
        }
    };
    let bytes = std::fs::read(&model).with_context(|| format!("reading {}", model.display()))?;
    let graph = import_model(&bytes, &ImportOptions::default())
        .with_context(|| format!("importing {}", model.display()))?;
    let tensors = inputs(&input_set).with_context(|| format!("reading {}", input_set.display()))?;
    let outcome = check_graph(&graph, &tensors, Expected::OnnxRuntime(expected), backend);
    if let Err(Failure::Mismatch(message)) = &outcome {
        ui::stat(&directory.display().to_string(), message);
    }
    tally.record(&outcome);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::MISMATCHED;
    use super::super::check::Failure;
    use super::super::export::{INPUTS, MODEL, write_case};
    use super::{Tally, check_case, check_cases, verdict};
    use ptxc_core::HostTensor;
    use ptxc_graph::{Op, UnaryOp};
    use ptxc_onnx::tensor_to_bytes;
    use ptxc_testkit::generate::{Input, Node, Operand, Recipe, Source};
    use ptxc_testkit::reference::cache_directory;
    use std::path::{Path, PathBuf};
    use std::process::ExitCode;

    fn vector(values: Vec<f32>) -> HostTensor {
        HostTensor::vector(values.into())
    }

    /// A case applying `op` to `[1, -2]`, written under `root` as `name`.
    fn case_of(root: &Path, name: &str, op: UnaryOp) -> PathBuf {
        let recipe = Recipe {
            inputs: vec![Input { tensor: vector(vec![1.0, -2.0]), source: Source::Moderate }],
            nodes: vec![Node { op: Op::Unary(op), operands: vec![Operand::Input(0)] }],
        };
        let directory = root.join(name);
        write_case(&directory, &recipe.build().unwrap()).unwrap();
        directory
    }

    /// A case negating `[1, -2]`, written under `root`.
    fn negation(root: &Path) -> PathBuf {
        case_of(root, "seed-0", UnaryOp::Neg)
    }

    /// `outputs` cached under `references` as a tool's, for `case`.
    fn cache(references: &Path, case: &Path, outputs: &[HostTensor]) {
        let model = case.join(MODEL);
        let tool =
            cache_directory(references, &model, &case.join(INPUTS)).unwrap().join("ort-test");
        std::fs::create_dir_all(&tool).unwrap();
        for (k, output) in outputs.iter().enumerate() {
            let bytes = tensor_to_bytes(output, &format!("y{k}")).unwrap();
            std::fs::write(tool.join(format!("output_{k}.pb")), bytes).unwrap();
        }
    }

    fn checked(cached: Option<Vec<f32>>) -> Tally {
        let dir = tempfile::tempdir().unwrap();
        let (cases, references) = (dir.path().join("cases"), dir.path().join("references"));
        let case = negation(&cases);
        if let Some(outputs) = cached {
            cache(&references, &case, &[vector(outputs)]);
        }
        let mut tally = Tally::default();
        check_case(&case, &references, None, &mut tally).unwrap();
        tally
    }

    #[test]
    fn a_case_matching_its_cached_outputs_agrees() {
        let tally = checked(Some(vec![-1.0, 2.0]));

        assert_eq!((tally.agreed, tally.mismatched, tally.unreferenced), (1, 0, 0));
    }

    #[test]
    fn a_case_differing_from_its_cached_outputs_mismatches() {
        let tally = checked(Some(vec![-1.0, 3.0]));

        assert_eq!((tally.agreed, tally.mismatched, tally.unreferenced), (0, 1, 0));
    }

    #[test]
    fn a_case_without_cached_outputs_is_counted_apart() {
        let tally = checked(None);

        assert_eq!((tally.agreed, tally.mismatched, tally.unreferenced), (0, 0, 1));
    }

    #[test]
    fn each_outcome_of_a_case_with_outputs_is_counted_apart() {
        let mut tally = Tally::default();

        tally.record(&Ok(()));
        tally.record(&Err(Failure::Rejected("division by zero".to_owned())));
        tally.record(&Err(Failure::Mismatch("differs".to_owned())));

        assert_eq!((tally.agreed, tally.rejected, tally.mismatched), (1, 1, 1));
    }

    #[test]
    fn every_directory_under_the_cases_is_a_case_and_nothing_else_is() {
        let dir = tempfile::tempdir().unwrap();
        let (cases, references) = (dir.path().join("cases"), dir.path().join("references"));
        let negated = negation(&cases);
        cache(&references, &negated, &[vector(vec![-1.0, 2.0])]);
        let _absolute = case_of(&cases, "seed-1", UnaryOp::Abs);
        std::fs::write(cases.join("notes.txt"), "not a case").unwrap();

        let tally = check_cases(&cases, &references, None).unwrap();

        assert_eq!((tally.agreed, tally.unreferenced, tally.mismatched), (1, 1, 0));
    }

    fn tally(agreed: u64, mismatched: u64) -> Tally {
        Tally { agreed, mismatched, ..Tally::default() }
    }

    #[test]
    fn cases_pass_unless_one_mismatched() {
        let directory = Path::new("cases");

        let passed = verdict(&tally(2, 0), directory).unwrap();
        let failed = verdict(&tally(0, 1), directory).unwrap();

        assert_eq!(passed, ExitCode::SUCCESS);
        assert_eq!(failed, ExitCode::from(MISMATCHED));
    }

    #[test]
    fn cases_none_of_which_has_outputs_are_an_error() {
        let unreferenced = Tally { unreferenced: 3, rejected: 1, ..Tally::default() };

        let error = verdict(&unreferenced, Path::new("cases")).unwrap_err();

        assert!(error.to_string().starts_with("no case under cases has ONNX Runtime outputs"));
    }
}
