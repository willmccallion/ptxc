//! Runs a graph through partitioning, lowering and the loop interpreter, and
//! checks it against `ptxc-interp`.

use ptxc_core::{DType, Evaluate, HostTensor};
use ptxc_graph::Graph;
use ptxc_interp::Interpreter;
use ptxc_testkit::compare::{Tolerance, compare};
use ptxc_testkit::lowered::run_lowered;

/// Checks the lowered graph agrees with the reference: within each output's
/// dtype tolerance, or failing wherever the reference fails.
pub(crate) fn check_against_reference(graph: Graph, inputs: &[HostTensor]) -> Result<(), String> {
    check_within(graph, inputs, Tolerance::for_dtype)
}

/// As `check_against_reference`, with `tolerance` giving each output's
/// tolerance from its element type.
pub(crate) fn check_within(
    graph: Graph,
    inputs: &[HostTensor],
    tolerance: impl Fn(DType) -> Tolerance,
) -> Result<(), String> {
    let expected = Interpreter::new(&graph).evaluate(inputs);
    let text = ptxc_graph::text::print(&graph);
    let actual = run_lowered(graph, inputs, ptxc_loop::run);
    match (expected, actual) {
        (Ok(expected), Ok(actual)) => {
            for (actual, expected) in actual.iter().zip(&expected) {
                compare(actual, expected, tolerance(expected.dtype()))
                    .map_err(|mismatch| format!("{mismatch}\n{text}\ninputs: {inputs:?}"))?;
            }
            Ok(())
        }
        (Err(_), Err(_)) => Ok(()),
        (expected, actual) => {
            Err(format!("reference {expected:?}, lowered {actual:?}\n{text}\ninputs: {inputs:?}"))
        }
    }
}
