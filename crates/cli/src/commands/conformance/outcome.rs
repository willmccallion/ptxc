//! One test case's outcome on a backend.

use crate::commands::agreement::agree;
use crate::commands::runner::Runner;
use crate::report::{Failure, Outcome, Reason, Worst};
use ptxc_core::HostTensor;
use ptxc_onnx::{ImportError, ImportErrorKind, ImportOptions, Unsupported, import_model};
use ptxc_testkit::compare::{self, Mismatch, Tolerance};
use ptxc_testkit::onnx_tests::{DataSet, LoadError, Runnable, TestCase};
use ptxc_testkit::sums::interpreted;

/// The node tests' own tolerance.
pub(super) const TOLERANCE: Tolerance = Tolerance::ONNX_SUITE;

/// The case's outcome: a pass only if every data set matches.
pub(super) fn of_case(case: &TestCase, runner: &Runner) -> Outcome {
    let data_sets = match case.data_sets() {
        Ok(data_sets) => data_sets,
        Err(error) => return model_unsupported(case).unwrap_or_else(|| of_load_error(&error)),
    };
    data_sets
        .iter()
        .enumerate()
        .map(|(index, data_set)| of_data_set(case, data_set, index, runner))
        .find(|outcome| *outcome != Outcome::Pass)
        .unwrap_or(Outcome::Pass)
}

fn of_data_set(case: &TestCase, data_set: &DataSet, index: usize, runner: &Runner) -> Outcome {
    let runnable = match case.import(data_set) {
        Ok(runnable) => runnable,
        Err(error) => return of_import_error(&error),
    };
    let actual = match runner.evaluate(&runnable.graph, &runnable.inputs) {
        Ok(actual) => actual,
        Err(message) => return Outcome::Fail { failure: Failure::Evaluate { message } },
    };
    if let Runner::Gpu(_) = runner
        && let Err(message) = agrees_with_loop_interpreter(&runnable, &actual)
    {
        return Outcome::Fail { failure: Failure::LoopInterpreter { data_set: index, message } };
    }
    if actual.len() != data_set.outputs.len() {
        let failure = Failure::OutputCount {
            data_set: index,
            expected: data_set.outputs.len(),
            actual: actual.len(),
        };
        return Outcome::Fail { failure };
    }
    for (output, (actual, expected)) in actual.iter().zip(&data_set.outputs).enumerate() {
        if let Err(mismatch) = compare::compare(actual, expected, TOLERANCE) {
            return Outcome::Fail { failure: of_mismatch(&mismatch, index, output) };
        }
    }
    Outcome::Pass
}

/// Whether the loop interpreter, running the lowered stages one by one,
/// gives `actual`, the GPU's outputs: exactly, or within tolerance where a
/// stage multiplies on tensor cores (D60).
fn agrees_with_loop_interpreter(runnable: &Runnable, actual: &[HostTensor]) -> Result<(), String> {
    let (lowered, rounding) = interpreted(runnable.graph.clone(), &runnable.inputs)
        .map_err(|error| format!("the loop interpreter failed: {error}"))?;
    agree(actual, &lowered, |dtype| rounding.tolerance(dtype))
}

/// Why the model alone is unsupported, if it is: a test of sequences or
/// optionals stores its data in protobufs other than tensors, which only
/// matters if ptxc supports the model.
fn model_unsupported(case: &TestCase) -> Option<Outcome> {
    let error = import_model(case.model(), &ImportOptions::default()).err()?;
    error.unsupported().map(|unsupported| Outcome::Unsupported { reason: reason(unsupported) })
}

/// An unsupported element type in the test data is unsupported; any other
/// load error is a failure.
pub(super) fn of_load_error(error: &LoadError) -> Outcome {
    if let LoadError::Tensor { source, .. } = error
        && let Some(unsupported) = source.unsupported()
    {
        return Outcome::Unsupported { reason: reason(unsupported) };
    }
    Outcome::Fail { failure: Failure::Load { message: error.to_string() } }
}

/// Unsupported constructs and shapes no input fixes are unsupported; any
/// other import error is a failure.
fn of_import_error(error: &ImportError) -> Outcome {
    if let Some(unsupported) = error.unsupported() {
        return Outcome::Unsupported { reason: reason(unsupported) };
    }
    if matches!(error.kind(), ImportErrorKind::NeedsConstant { .. }) {
        return Outcome::Unsupported { reason: Reason::ValueDependentShape };
    }
    Outcome::Fail { failure: Failure::Import { message: error.to_string() } }
}

fn reason(unsupported: &Unsupported) -> Reason {
    match unsupported {
        Unsupported::Op { domain, op } if domain.is_empty() => Reason::Op(op.clone()),
        Unsupported::Op { domain, op } => Reason::Op(format!("{domain}.{op}")),
        Unsupported::DType(dtype) => Reason::DType(dtype.clone()),
        Unsupported::Attribute { op, attribute } => {
            Reason::Attribute { op: op.clone(), attribute: attribute.clone() }
        }
        Unsupported::Opset(version) => Reason::Opset(*version),
        Unsupported::ControlFlow(op) => Reason::ControlFlow(op.clone()),
        Unsupported::DynamicShape(dim) => Reason::DynamicShape(dim.clone()),
    }
}

fn of_mismatch(mismatch: &Mismatch, data_set: usize, output: usize) -> Failure {
    let (worst_absolute, worst_relative) = match mismatch {
        Mismatch::Values(report) => {
            (Some(worst(&report.worst_absolute)), Some(worst(&report.worst_relative)))
        }
        Mismatch::DType { .. } | Mismatch::Shape { .. } | Mismatch::Bounds(_) => (None, None),
    };
    Failure::Mismatch {
        data_set,
        output,
        message: mismatch.to_string(),
        worst_absolute,
        worst_relative,
    }
}

fn worst(worst: &compare::Worst) -> Worst {
    Worst {
        position: worst.position.0.clone(),
        actual: worst.actual,
        expected: worst.expected,
        error: worst.error,
    }
}

#[cfg(test)]
mod tests {
    use super::agrees_with_loop_interpreter;
    use ptxc_core::{DType, Dim, HostTensor, Shape};
    use ptxc_graph::{GraphBuilder, Layout, Op, TensorType, UnaryOp};
    use ptxc_testkit::onnx_tests::Runnable;

    fn vector(values: Vec<f32>) -> HostTensor {
        HostTensor::vector(values.into())
    }

    /// Negating `[1, -2]`.
    fn negation() -> Runnable {
        let mut builder = GraphBuilder::new();
        let shape = Shape::new(vec![Dim::new(2)]).unwrap();
        let x = builder.add_input(TensorType::new(DType::F32, shape, Layout::RowMajor));
        let y = builder.add_node(Op::Unary(UnaryOp::Neg), vec![x]).unwrap();
        Runnable { graph: builder.finish(y).unwrap(), inputs: vec![vector(vec![1.0, -2.0])] }
    }

    #[test]
    fn a_gpu_result_the_loop_interpreter_does_not_give_is_refused() {
        let runnable = negation();

        let given = agrees_with_loop_interpreter(&runnable, &[vector(vec![-1.0, 2.0])]);
        let other = agrees_with_loop_interpreter(&runnable, &[vector(vec![-1.0, 3.0])]);

        assert_eq!(given, Ok(()));
        assert!(other.is_err());
    }
}
