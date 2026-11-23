//! `Stage` and `LoweredGroup`: compute definitions bound to the tensors they
//! read and write, and the builder lowering uses to make a stage.

use ptxc_core::{DType, Dim, IndexVec, Shape, define_index};
use ptxc_graph::{Graph, Layout, TensorType, ValueId};
use ptxc_loop::{
    ComputeBuilder, ComputeDef, ComputeError, ComputeName, ExprArena, ExprId, IterVarId,
    TensorExpr, TensorName, TensorParamId, VarName,
};
use std::sync::Arc;

define_index!(
    /// A buffer joining two stages of one group.
    pub IntermediateId
);

/// The tensor bound to one of a stage's inputs or outputs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Binding {
    /// A graph value.
    Value(ValueId),
    /// One of the group's intermediate buffers.
    Intermediate(IntermediateId),
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

/// A group's stages, in the order they run, and the intermediate buffers
/// joining them. A stage reads graph values computed by earlier groups and
/// intermediates written by earlier stages.
#[derive(Clone, Debug, Default)]
pub struct LoweredGroup {
    intermediates: IndexVec<IntermediateId, TensorType>,
    stages: Vec<Stage>,
}

impl LoweredGroup {
    pub(crate) const fn new(
        intermediates: IndexVec<IntermediateId, TensorType>,
        stages: Vec<Stage>,
    ) -> Self {
        Self { intermediates, stages }
    }

    /// The type of each intermediate buffer.
    pub const fn intermediates(&self) -> &IndexVec<IntermediateId, TensorType> {
        &self.intermediates
    }

    /// The stages, in the order they run.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }
}

/// Builds a group stage by stage, declaring each intermediate before the
/// stages that write and read it.
pub(crate) struct GroupBuilder<'g> {
    graph: &'g Graph,
    intermediates: IndexVec<IntermediateId, TensorType>,
    stages: Vec<Stage>,
}

impl<'g> GroupBuilder<'g> {
    pub(crate) const fn new(graph: &'g Graph) -> Self {
        Self { graph, intermediates: IndexVec::new(), stages: Vec::new() }
    }

    /// A new f32 intermediate of `shape`: statistics stay in f32.
    pub(crate) fn intermediate(&mut self, shape: &Shape) -> IntermediateId {
        self.intermediates.push(TensorType::new(DType::F32, shape.clone(), Layout::RowMajor))
    }

    /// A stage over `shape` that may read the intermediates declared so far.
    pub(crate) fn stage(&self, name: ComputeName, shape: &Shape) -> StageBuilder<'g> {
        StageBuilder::new(self.graph, self.intermediates.clone(), name, shape)
    }

    pub(crate) fn push(&mut self, stage: Stage) {
        self.stages.push(stage);
    }

    pub(crate) fn finish(self) -> LoweredGroup {
        LoweredGroup::new(self.intermediates, self.stages)
    }
}

/// An operand loaded at an element of the stage's output.
#[derive(Clone, Debug)]
pub(crate) struct Operand {
    pub(crate) expr: TensorExpr,
    pub(crate) dtype: DType,
}

/// Builds a stage whose spatial axes are an output shape's dimensions.
pub(crate) struct StageBuilder<'g> {
    graph: &'g Graph,
    intermediates: IndexVec<IntermediateId, TensorType>,
    compute: ComputeBuilder,
    inputs: IndexVec<TensorParamId, Binding>,
    outputs: Vec<Binding>,
    spatial: Vec<ExprId>,
    reduce_axes: usize,
}

impl<'g> StageBuilder<'g> {
    /// A stage named `name` with one spatial axis per dimension of `shape`,
    /// reading graph values and `intermediates`.
    pub(crate) fn new(
        graph: &'g Graph,
        intermediates: IndexVec<IntermediateId, TensorType>,
        name: ComputeName,
        shape: &Shape,
    ) -> Self {
        let mut compute = ComputeBuilder::new(name);
        let spatial = shape
            .dims()
            .iter()
            .enumerate()
            .map(|(position, &extent)| {
                compute.spatial(VarName::new(format!("i{position}")), extent).1
            })
            .collect();
        let (inputs, outputs) = (IndexVec::new(), Vec::new());
        Self { graph, intermediates, compute, inputs, outputs, spatial, reduce_axes: 0 }
    }

    /// The spatial axes' index variables, one per output dimension.
    pub(crate) fn spatial(&self) -> &[ExprId] {
        &self.spatial
    }

    /// A new reduction axis over `extent`, named `k<n>`.
    pub(crate) fn reduce_axis(&mut self, extent: Dim) -> (IterVarId, ExprId) {
        let name = VarName::new(format!("k{}", self.reduce_axes));
        self.reduce_axes += 1;
        self.compute.reduce_axis(name, extent)
    }

    /// The index expression `value`.
    pub(crate) fn constant(&mut self, value: i64) -> ExprId {
        self.compute.exprs().constant(value)
    }

    /// The arena the stage's index expressions are built in.
    pub(crate) const fn exprs(&mut self) -> &mut ExprArena {
        self.compute.exprs()
    }

    fn type_of(&self, binding: Binding) -> &TensorType {
        match binding {
            Binding::Value(value) => self.graph.value(value).ty(),
            Binding::Intermediate(id) => &self.intermediates[id],
        }
    }

    /// `binding`'s tensor, added as an input the first time it is read.
    fn tensor(&mut self, binding: Binding) -> TensorParamId {
        if let Some((tensor, _)) =
            self.inputs.iter_enumerated().find(|&(_, &bound)| bound == binding)
        {
            return tensor;
        }
        let ty = self.type_of(binding).clone();
        let name = TensorName::new(format!("in{}", self.inputs.iter().len()));
        let tensor = self.compute.input(name, ty.dtype(), ty.shape().clone());
        let _bound = self.inputs.push(binding);
        tensor
    }

    /// `binding` read at `index`, one index per axis.
    pub(crate) fn load(&mut self, binding: Binding, index: Vec<ExprId>) -> Operand {
        let tensor = self.tensor(binding);
        Operand { expr: TensorExpr::Load { tensor, index }, dtype: self.type_of(binding).dtype() }
    }

    /// An index variable holding `binding`'s integer at `index`, as a gather
    /// reads its indices.
    pub(crate) fn loaded(&mut self, name: &str, binding: Binding, index: Vec<ExprId>) -> ExprId {
        let tensor = self.tensor(binding);
        self.compute.loaded_index(VarName::new(name), tensor, index)
    }

    /// `value` read at the output element, with ONNX's multidirectional
    /// broadcasting: trailing axes align, and an axis of extent 1 is read
    /// at 0.
    pub(crate) fn broadcast(&mut self, value: ValueId) -> Operand {
        let zero = self.constant(0);
        let dims = self.graph.value(value).ty().shape().dims();
        let mut index: Vec<ExprId> = dims
            .iter()
            .rev()
            .zip(self.spatial.iter().rev())
            .map(|(dim, &axis)| if dim.get() == 1 { zero } else { axis })
            .collect();
        index.reverse();
        self.load(Binding::Value(value), index)
    }

    /// Adds an output computing `binding` as `expr`.
    pub(crate) fn output(&mut self, binding: Binding, expr: TensorExpr) {
        let name = TensorName::new(format!("out{}", self.outputs.len()));
        self.compute.output(name, expr);
        self.outputs.push(binding);
    }

    /// The stage, if its definition is well-formed.
    pub(crate) fn finish(self) -> Result<Stage, ComputeError> {
        let compute = Arc::new(self.compute.finish()?);
        Ok(Stage { compute, inputs: self.inputs, outputs: self.outputs })
    }
}
