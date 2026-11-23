//! One fuzz case checked across executors: the lowered graph on the loop
//! interpreter against the reference within tolerance, and a backend, such
//! as the GPU, against the loop interpreter exactly.

use ptxc_core::{DType, ElementCount, Evaluate, HostTensor};
use ptxc_graph::Graph;
use ptxc_interp::Interpreter;
use ptxc_testkit::compare::{Tolerance, compare};
use ptxc_testkit::generate::{Case, Recipe};
use ptxc_testkit::lowered::run_lowered;

/// An executor checked beyond the interpreters.
pub(super) type Backend<'a> = dyn Fn(&Graph, &[HostTensor]) -> Result<Vec<HostTensor>, String> + 'a;

/// Whether every executor agrees on `recipe`'s case, and if not, how they
/// differ.
pub(super) fn check(recipe: &Recipe, backend: Option<&Backend<'_>>) -> Result<(), String> {
    let Case { graph, inputs } =
        recipe.build().ok_or_else(|| "the recipe does not build".to_owned())?;
    let reference = Interpreter::new(&graph)
        .evaluate(&inputs)
        .map_err(|error| format!("the reference failed: {error}"))?;
    let lowered = run_lowered(graph.clone(), &inputs, ptxc_loop::run)
        .map_err(|error| format!("the loop interpreter failed: {error}"))?;
    let length = longest(&inputs);
    agree(&lowered, &reference, |dtype| Tolerance::for_dtype(dtype).for_reduction(length))
        .map_err(|message| format!("lowered against the reference: {message}"))?;
    let Some(backend) = backend else { return Ok(()) };
    let outputs =
        backend(&graph, &inputs).map_err(|error| format!("the backend failed: {error}"))?;
    agree(&outputs, &lowered, |_| Tolerance::Exact)
        .map_err(|message| format!("the backend against the loop interpreter: {message}"))
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
