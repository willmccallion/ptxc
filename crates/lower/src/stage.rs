//! `Stage` and `LoweredGroup`: compute definitions bound to the tensors they
//! read and write.

use ptxc_core::{HostTensor, IndexVec, define_index};
use ptxc_graph::{TensorType, ValueId};
use ptxc_loop::{ComputeDef, TensorParamId};
use std::sync::Arc;

define_index!(
    /// A buffer joining two stages of one group.
    pub IntermediateId
);

define_index!(
    /// A constant table a group's stages read, built from an op's
    /// attributes, such as per-channel quantization scales (D46).
    pub TableId
);

/// The tensor bound to one of a stage's inputs or outputs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Binding {
    /// A graph value.
    Value(ValueId),
    /// One of the group's intermediate buffers.
    Intermediate(IntermediateId),
    /// One of the group's constant tables.
    Table(TableId),
}

/// One compute definition and the tensor bound to each of its inputs and
/// outputs. Each stage is one kernel in M2.
#[derive(Clone, Debug)]
pub struct Stage {
    compute: Arc<ComputeDef>,
    inputs: IndexVec<TensorParamId, Binding>,
    outputs: Vec<Binding>,
}

impl Stage {
    pub(crate) const fn new(
        compute: Arc<ComputeDef>,
        inputs: IndexVec<TensorParamId, Binding>,
        outputs: Vec<Binding>,
    ) -> Self {
        Self { compute, inputs, outputs }
    }

    /// What the stage computes.
    pub const fn compute(&self) -> &Arc<ComputeDef> {
        &self.compute
    }

    /// The tensor bound to each of the definition's inputs.
    pub const fn inputs(&self) -> &IndexVec<TensorParamId, Binding> {
        &self.inputs
    }

    /// The tensor each of the definition's outputs computes, in order.
    pub fn outputs(&self) -> &[Binding] {
        &self.outputs
    }
}

/// A group's stages, the intermediate buffers joining them, and the
/// constant tables they read.
///
/// The stages run in order. Each reads graph values computed by earlier
/// groups and intermediates written by earlier stages.
#[derive(Clone, Debug, Default)]
pub struct LoweredGroup {
    intermediates: IndexVec<IntermediateId, TensorType>,
    tables: IndexVec<TableId, HostTensor>,
    stages: Vec<Stage>,
}

impl LoweredGroup {
    pub(crate) const fn new(
        intermediates: IndexVec<IntermediateId, TensorType>,
        tables: IndexVec<TableId, HostTensor>,
        stages: Vec<Stage>,
    ) -> Self {
        Self { intermediates, tables, stages }
    }

    /// The type of each intermediate buffer.
    pub const fn intermediates(&self) -> &IndexVec<IntermediateId, TensorType> {
        &self.intermediates
    }

    /// The constant tables, uploaded before the group runs.
    pub const fn tables(&self) -> &IndexVec<TableId, HostTensor> {
        &self.tables
    }

    /// The stages, in the order they run.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }
}
