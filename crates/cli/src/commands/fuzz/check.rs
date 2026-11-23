//! One fuzz case checked across executors: the lowered graph on the loop
//! interpreter against the reference within tolerance, when the recipe
//! allows it, and a backend, such as the GPU, against the loop interpreter
//! exactly.

use ptxc_core::{DType, ElementCount, Evaluate, HostTensor};
use ptxc_graph::Graph;
use ptxc_interp::Interpreter;
use ptxc_loop::RunError;
use ptxc_testkit::compare::{Tolerance, compare};
use ptxc_testkit::generate::{Case, Recipe};
use ptxc_testkit::lowered::{LoweredError, run_lowered};

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

/// Whether every executor agrees on `recipe`'s case, and if not, why.
pub(super) fn check(recipe: &Recipe, backend: Option<&Backend<'_>>) -> Result<(), Failure> {
    let Case { graph, inputs } =
        recipe.build().ok_or_else(|| mismatch("the recipe does not build".to_owned()))?;
    let lowered = run_lowered(graph.clone(), &inputs, ptxc_loop::run).map_err(lowering_failure)?;
    if recipe.compares_with_reference() {
        let reference = Interpreter::new(&graph)
            .evaluate(&inputs)
            .map_err(|error| mismatch(format!("the reference failed: {error}")))?;
        let length = longest(&inputs);
        agree(&lowered, &reference, |dtype| Tolerance::for_dtype(dtype).for_reduction(length))
            .map_err(|message| mismatch(format!("lowered against the reference: {message}")))?;
    }
    let Some(backend) = backend else { return Ok(()) };
    let outputs = backend(&graph, &inputs)
        .map_err(|error| mismatch(format!("the backend failed: {error}")))?;
    agree(&outputs, &lowered, |_| Tolerance::Exact)
        .map_err(|message| mismatch(format!("the backend against the loop interpreter: {message}")))
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

/// Whether `actual` matches `expected` output by output, each within
/// `tolerance` of its element type.
fn agree(
    actual: &[HostTensor],
    expected: &[HostTensor],
    tolerance: impl Fn(DType) -> Tolerance,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("{} outputs, expected {}", actual.len(), expected.len()));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        compare(actual, expected, tolerance(expected.dtype()))
            .map_err(|mismatch| format!("output {index}: {mismatch}"))?;
    }
    Ok(())
}
