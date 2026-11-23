//! Runs a graph through partitioning, lowering and the loop interpreter, and
//! checks it against `ptxc-interp`.

use ptxc_core::{Evaluate, HostTensor};
use ptxc_graph::{Graph, Partitioned, ValueDef, ValueId};
use ptxc_interp::Interpreter;
use ptxc_loop::{RunError, Schedule};
use ptxc_target::{SmVersion, preset};
use ptxc_testkit::compare::{Tolerance, compare};
use std::collections::HashMap;
use std::sync::Arc;

/// `graph`'s outputs for `inputs`, each group lowered and each stage run
/// under its default schedule.
pub(crate) fn run_lowered(
    graph: Graph,
    inputs: &[HostTensor],
) -> Result<Vec<HostTensor>, RunError> {
    let partitioned = Partitioned::one_node_per_group(graph).unwrap();
    let groups = ptxc_lower::lower(&partitioned).unwrap();
    let graph = partitioned.graph();
    let target = preset(SmVersion::SM_89).unwrap();
    let mut computed: HashMap<ValueId, HostTensor> =
        graph.inputs().iter().copied().zip(inputs.iter().cloned()).collect();
    for stage in groups.iter().flatten() {
        let arguments: Vec<HostTensor> =
            stage.inputs().iter().map(|&value| lookup(graph, &computed, value)).collect();
        let schedule = Schedule::new(Arc::clone(stage.compute()));
        let kernel = schedule.validate(&target).unwrap().lower();
        for (&value, flat) in stage.outputs().iter().zip(ptxc_loop::run(&kernel, &arguments)?) {
            let shape = graph.value(value).ty().shape().clone();
            let _new = computed.insert(value, HostTensor::new(shape, flat.data().clone()).unwrap());
        }
    }
    Ok(graph.outputs().iter().map(|&value| lookup(graph, &computed, value)).collect())
}

fn lookup(graph: &Graph, computed: &HashMap<ValueId, HostTensor>, value: ValueId) -> HostTensor {
    match graph.value(value).def() {
        ValueDef::Constant(constant) => graph.constant(constant).clone(),
        ValueDef::Input(_) | ValueDef::Output { .. } => computed[&value].clone(),
    }
}

/// Checks the lowered graph agrees with the reference: within each output's
/// dtype tolerance, or failing wherever the reference fails.
pub(crate) fn check_against_reference(graph: Graph, inputs: &[HostTensor]) -> Result<(), String> {
    let expected = Interpreter::new(&graph).evaluate(inputs);
    let text = ptxc_graph::text::print(&graph);
    let actual = run_lowered(graph, inputs);
    match (expected, actual) {
        (Ok(expected), Ok(actual)) => {
            for (actual, expected) in actual.iter().zip(&expected) {
                compare(actual, expected, Tolerance::for_dtype(expected.dtype()))
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
