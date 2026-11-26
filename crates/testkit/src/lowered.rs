//! Runs a graph as ptxc compiles it before codegen, stage by stage.
//!
//! Each group lowered and its products split where the compiler splits
//! them (D75), and each stage put under its default schedule and run by an
//! executor the caller gives, such as the loop interpreter or the GPU.

use ptxc_core::{HostTensor, Idx, TensorError};
use ptxc_graph::{CycleError, Graph, Partitioned, ValueDef, ValueId};
use ptxc_loop::{ComputeDef, LoweredKernel, ScheduleError};
use ptxc_lower::{Binding, IntermediateId, LowerError, LoweredGroup};
use ptxc_target::{SmVersion, UnsupportedSm, preset};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Why a lowered graph did not run.
#[derive(Debug, Error)]
pub enum LoweredError<E> {
    /// The graph has a cycle.
    #[error(transparent)]
    Partition(#[from] CycleError),
    /// Lowering refused a group.
    #[error(transparent)]
    Lower(#[from] LowerError),
    /// The default schedule refused a stage.
    #[error(transparent)]
    Schedule(#[from] ScheduleError),
    /// The target has no preset.
    #[error(transparent)]
    Target(#[from] UnsupportedSm),
    /// A stage read a tensor no earlier stage wrote, or wrote a table.
    #[error("a stage's binding {0:?} is out of order")]
    Binding(Binding),
    /// A stage's output does not fit the tensor it binds.
    #[error(transparent)]
    Shape(#[from] TensorError),
    /// A stage's executor failed.
    #[error("a stage failed: {0}")]
    Stage(E),
}

/// `graph`'s outputs for `inputs`, each node its own group, each stage's
/// kernel run by `run_stage` on its arguments, returning its outputs flat
/// in its output order.
pub fn run_lowered<E>(
    graph: Graph,
    inputs: &[HostTensor],
    run_stage: impl FnMut(&LoweredKernel, &[HostTensor]) -> Result<Vec<HostTensor>, E>,
) -> Result<Vec<HostTensor>, LoweredError<E>> {
    run_partitioned(&Partitioned::one_node_per_group(graph)?, inputs, run_stage)
}

/// `partitioned`'s outputs for `inputs`, as `run_lowered` runs a graph's,
/// each group lowered as `partitioned` groups its nodes.
pub fn run_partitioned<E>(
    partitioned: &Partitioned,
    inputs: &[HostTensor],
    mut run_stage: impl FnMut(&LoweredKernel, &[HostTensor]) -> Result<Vec<HostTensor>, E>,
) -> Result<Vec<HostTensor>, LoweredError<E>> {
    let groups = ptxc_lower::lower(partitioned)?;
    let graph = partitioned.graph();
    let target = preset(SmVersion::SM_89)?;
    let mut tensors = Tensors {
        graph,
        values: graph.inputs().iter().copied().zip(inputs.iter().cloned()).collect(),
        intermediates: HashMap::new(),
        tables: Vec::new(),
    };
    for group in groups {
        let split = |compute: &Arc<ComputeDef>| ptxc_schedule::split_k(compute, &target);
        let group = group.with_split_reductions(split)?;
        tensors.intermediates.clear();
        tensors.tables = group.tables().iter().cloned().collect();
        for stage in group.stages() {
            let arguments = stage
                .inputs()
                .iter()
                .map(|&binding| tensors.read(binding))
                .collect::<Result<Vec<_>, _>>()?;
            let schedule = ptxc_schedule::default_schedule(Arc::clone(stage.compute()), &target)?;
            let outputs = run_stage(&schedule.lower(), &arguments).map_err(LoweredError::Stage)?;
            for (&binding, flat) in stage.outputs().iter().zip(outputs) {
                tensors.write(&group, binding, flat)?;
            }
        }
    }
    graph.outputs().iter().map(|&value| tensors.read(Binding::Value(value))).collect()
}

/// Every tensor computed so far: graph values, and the current group's
/// intermediates and tables.
struct Tensors<'g> {
    graph: &'g Graph,
    values: HashMap<ValueId, HostTensor>,
    intermediates: HashMap<IntermediateId, HostTensor>,
    tables: Vec<HostTensor>,
}

impl Tensors<'_> {
    fn read<E>(&self, binding: Binding) -> Result<HostTensor, LoweredError<E>> {
        let found = match binding {
            Binding::Value(value) => match self.graph.value(value).def() {
                ValueDef::Constant(constant) => Some(self.graph.constant(constant)),
                ValueDef::Input(_) | ValueDef::Output { .. } => self.values.get(&value),
            },
            Binding::Intermediate(id) => self.intermediates.get(&id),
            Binding::Table(id) => self.tables.get(id.index()),
        };
        found.cloned().ok_or(LoweredError::Binding(binding))
    }

    /// Stores a stage's flat output with the shape of the tensor it binds.
    fn write<E>(
        &mut self,
        group: &LoweredGroup,
        binding: Binding,
        flat: HostTensor,
    ) -> Result<(), LoweredError<E>> {
        let (_, data) = flat.into_parts();
        match binding {
            Binding::Value(value) => {
                let shape = self.graph.value(value).ty().shape().clone();
                let _new = self.values.insert(value, HostTensor::new(shape, data)?);
            }
            Binding::Intermediate(id) => {
                let shape = group.intermediates()[id].shape().clone();
                let _new = self.intermediates.insert(id, HostTensor::new(shape, data)?);
            }
            Binding::Table(_) => return Err(LoweredError::Binding(binding)),
        }
        Ok(())
    }
}
