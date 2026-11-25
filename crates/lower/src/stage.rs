//! `Stage` and `LoweredGroup`: compute definitions bound to the tensors they
//! read and write.

use ptxc_core::{HostTensor, IndexVec, define_index};
use ptxc_graph::{Layout, TensorType, ValueId};
use ptxc_loop::{CombineInput, ComputeDef, Rfactored, TensorParamId};
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
/// outputs. Each stage is one kernel.
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

    /// The stage's partial and combine stages from `rfactored`, the
    /// partials held in `partials`.
    fn rfactored(&self, rfactored: Rfactored, partials: IntermediateId) -> [Self; 2] {
        let Rfactored { partial, partial_inputs, combine, combine_inputs } = rfactored;
        let partial_reads = partial_inputs.iter().map(|&original| self.inputs[original]).collect();
        let combine_reads = combine_inputs
            .iter()
            .map(|&input| match input {
                CombineInput::Partials => Binding::Intermediate(partials),
                CombineInput::Original(original) => self.inputs[original],
            })
            .collect();
        let partials = vec![Binding::Intermediate(partials)];
        [
            Self::new(Arc::new(partial), partial_reads, partials),
            Self::new(Arc::new(combine), combine_reads, self.outputs.clone()),
        ]
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

    /// The group with each stage whose definition `split` rfactors replaced
    /// by its partial and combine stages, joined by a new intermediate that
    /// holds the partials (D75).
    pub fn with_split_reductions<E>(
        self,
        mut split: impl FnMut(&Arc<ComputeDef>) -> Result<Option<Rfactored>, E>,
    ) -> Result<Self, E> {
        let Self { mut intermediates, tables, stages } = self;
        let mut split_stages = Vec::with_capacity(stages.len());
        for stage in stages {
            let Some(rfactored) = split(stage.compute())? else {
                split_stages.push(stage);
                continue;
            };
            let held = rfactored.partials();
            let ty = TensorType::new(held.dtype, held.shape.clone(), Layout::RowMajor);
            split_stages.extend(stage.rfactored(rfactored, intermediates.push(ty)));
        }
        Ok(Self { intermediates, tables, stages: split_stages })
    }
}
