//! One fuzz case checked across executors: the lowered graph on the loop
//! interpreter against an independent result within tolerance, where there
//! is one, and a backend, such as the GPU, against the loop interpreter
//! exactly, or within tolerance where it multiplies on tensor cores (D60).

use crate::commands::agreement::agree;
use ptxc_core::{ElementCount, Evaluate, HostTensor};
use ptxc_graph::Graph;
use ptxc_interp::Interpreter;
use ptxc_loop::RunError;
use ptxc_testkit::compare::Tolerance;
use ptxc_testkit::generate::{Case, Recipe};
use ptxc_testkit::lowered::LoweredError;
use ptxc_testkit::sums::{SumRounding, interpreted};

/// An executor checked beyond the interpreters.
pub(super) type Backend<'a> = dyn Fn(&Graph, &[HostTensor]) -> Result<Vec<HostTensor>, String> + 'a;

/// Why a case's executors did not agree.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Failure {
    /// The loop interpreter refused the inputs' values, such as an index
    /// out of range or a division by zero, so nothing is compared.
    Rejected(String),
    /// The executors disagreed, or one failed, as described.
    Mismatch(String),
}

/// The independent result the loop interpreter is compared with, within
/// tolerance.
pub(super) enum Expected {
    /// None: the graph's ops are discontinuous or its inputs extreme, so
    /// only the backend is compared, exactly.
    Nothing,
    /// The reference interpreter's outputs.
    Reference,
    /// ONNX Runtime's cached outputs.
    OnnxRuntime(Vec<HostTensor>),
}

/// Whether every executor agrees on `recipe`'s case, and if not, why.
pub(super) fn check(recipe: &Recipe, backend: Option<&Backend<'_>>) -> Result<(), Failure> {
    let Case { graph, inputs } =
        recipe.build().ok_or_else(|| mismatch("the recipe does not build".to_owned()))?;
    let expected =
        if recipe.compares_with_reference() { Expected::Reference } else { Expected::Nothing };
    check_graph(&graph, &inputs, expected, backend)
}

/// Whether every executor agrees on `graph` for `inputs`, and with
/// `expected`, and if not, why.
pub(super) fn check_graph(
    graph: &Graph,
    inputs: &[HostTensor],
    expected: Expected,
    backend: Option<&Backend<'_>>,
) -> Result<(), Failure> {
    let (lowered, rounding) = interpreted(graph.clone(), inputs).map_err(lowering_failure)?;
    let continuous = !matches!(expected, Expected::Nothing);
    if let Some((source, expected)) = independent(graph, inputs, expected)? {
        let length = longest(inputs);
        agree(&lowered, &expected, |dtype| Tolerance::for_dtype(dtype).for_reduction(length))
            .map_err(|message| mismatch(format!("lowered against {source}: {message}")))?;
    }
    let Some(backend) = backend else { return Ok(()) };
    let outputs =
        backend(graph, inputs).map_err(|error| mismatch(format!("the backend failed: {error}")))?;
    if !compares_backend(rounding, continuous) {
        return Ok(());
    }
    agree(&outputs, &lowered, |dtype| rounding.tolerance(dtype))
        .map_err(|message| mismatch(format!("the backend against the loop interpreter: {message}")))
}

/// The outputs `expected` names, and where they came from.
fn independent(
    graph: &Graph,
    inputs: &[HostTensor],
    expected: Expected,
) -> Result<Option<(&'static str, Vec<HostTensor>)>, Failure> {
    Ok(match expected {
        Expected::Nothing => None,
        Expected::Reference => {
            let outputs = Interpreter::new(graph)
                .evaluate(inputs)
                .map_err(|error| mismatch(format!("the reference failed: {error}")))?;
            Some(("the reference", outputs))
        }
        Expected::OnnxRuntime(outputs) => Some(("ONNX Runtime", outputs)),
    })
}

/// Whether a backend's outputs are compared with the loop interpreter's:
/// unless the case is discontinuous and a tensor core's last-bit difference
/// in a sum could flip a later op.
pub(super) const fn compares_backend(rounding: SumRounding, continuous: bool) -> bool {
    continuous || matches!(rounding, SumRounding::Interpreted)
}

const fn mismatch(message: String) -> Failure {
    Failure::Mismatch(message)
}

/// A rejection if a stage refused its inputs' values, else a mismatch.
fn lowering_failure(error: LoweredError<RunError>) -> Failure {
    match error {
        LoweredError::Stage(
            refusal @ (RunError::OutOfBounds { .. } | RunError::Index | RunError::DivisionByZero),
        ) => Failure::Rejected(refusal.to_string()),
        other => mismatch(format!("the loop interpreter failed: {other}")),
    }
}

/// The largest input's element count, which bounds any reduction's length.
fn longest(inputs: &[HostTensor]) -> ElementCount {
    let most = inputs.iter().map(|input| input.shape().element_count().get()).max();
    ElementCount::new(most.unwrap_or(1).max(1))
}
