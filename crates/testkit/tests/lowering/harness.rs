//! Runs a graph through partitioning, lowering and the loop interpreter, and
//! checks it against `ptxc-interp`.

use ptxc_core::{DType, Evaluate, HostTensor};
use ptxc_graph::{Graph, Partitioned, ValueDef, ValueId};
use ptxc_interp::Interpreter;
use ptxc_loop::{RunError, Schedule};
use ptxc_lower::{Binding, IntermediateId, LoweredGroup};
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
    let mut tensors = Tensors {
        graph,
        values: graph.inputs().iter().copied().zip(inputs.iter().cloned()).collect(),
        intermediates: HashMap::new(),
    };
    for group in &groups {
        tensors.intermediates.clear();
        for stage in group.stages() {
            let arguments: Vec<HostTensor> =
                stage.inputs().iter().map(|&binding| tensors.read(binding)).collect();
            let schedule = Schedule::new(Arc::clone(stage.compute()));
            let kernel = schedule.validate(&target).unwrap().lower();
            let outputs = ptxc_loop::run(&kernel, &arguments)?;
            for (&binding, flat) in stage.outputs().iter().zip(outputs) {
                tensors.write(group, binding, &flat);
            }
        }
    }
    Ok(graph.outputs().iter().map(|&value| tensors.read(Binding::Value(value))).collect())
}

/// Every tensor computed so far: graph values, and the current group's
/// intermediates.
struct Tensors<'g> {
    graph: &'g Graph,
    values: HashMap<ValueId, HostTensor>,
    intermediates: HashMap<IntermediateId, HostTensor>,
}

impl Tensors<'_> {
    fn read(&self, binding: Binding) -> HostTensor {
        match binding {
            Binding::Value(value) => match self.graph.value(value).def() {
                ValueDef::Constant(constant) => self.graph.constant(constant).clone(),
                ValueDef::Input(_) | ValueDef::Output { .. } => self.values[&value].clone(),
            },
            Binding::Intermediate(id) => self.intermediates[&id].clone(),
        }
    }

    /// Stores a stage's flat output with the shape of the tensor it binds.
    fn write(&mut self, group: &LoweredGroup, binding: Binding, flat: &HostTensor) {
        match binding {
            Binding::Value(value) => {
                let shape = self.graph.value(value).ty().shape().clone();
                let _new =
                    self.values.insert(value, HostTensor::new(shape, flat.data().clone()).unwrap());
            }
            Binding::Intermediate(id) => {
                let shape = group.intermediates()[id].shape().clone();
                let tensor = HostTensor::new(shape, flat.data().clone()).unwrap();
                let _new = self.intermediates.insert(id, tensor);
            }
        }
    }
}

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
    let actual = run_lowered(graph, inputs);
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
