//! The builders lowering makes groups and stages with.

use crate::stage::{Binding, IntermediateId, LoweredGroup, Stage, TableId};
use ptxc_core::{DType, Dim, HostTensor, IndexVec, Shape};
use ptxc_graph::{Graph, Layout, TensorType, ValueId};
use ptxc_loop::{
    ComputeBuilder, ComputeError, ComputeName, ExprArena, ExprId, IterVarId, TensorExpr,
    TensorName, TensorParamId, VarName,
};
use std::sync::Arc;

/// The types of a group's own tensors that a stage may read.
#[derive(Clone)]
pub(crate) struct Locals {
    intermediates: IndexVec<IntermediateId, TensorType>,
    tables: IndexVec<TableId, TensorType>,
}

impl Locals {
    /// No intermediates or tables, as a single stage has.
    pub(crate) const fn empty() -> Self {
        Self { intermediates: IndexVec::new(), tables: IndexVec::new() }
    }
}

/// Builds a group stage by stage, declaring each intermediate and table
/// before the stages that read it.
pub(crate) struct GroupBuilder<'g> {
    graph: &'g Graph,
    locals: Locals,
    tables: IndexVec<TableId, HostTensor>,
    stages: Vec<Stage>,
}

impl<'g> GroupBuilder<'g> {
    pub(crate) const fn new(graph: &'g Graph) -> Self {
        Self { graph, locals: Locals::empty(), tables: IndexVec::new(), stages: Vec::new() }
    }

    /// A new f32 intermediate of `shape`: statistics stay in f32.
    pub(crate) fn intermediate(&mut self, shape: &Shape) -> IntermediateId {
        self.intermediate_of(DType::F32, shape)
    }

    /// A new intermediate of `dtype` and `shape`.
    pub(crate) fn intermediate_of(&mut self, dtype: DType, shape: &Shape) -> IntermediateId {
        let ty = TensorType::new(dtype, shape.clone(), Layout::RowMajor);
        self.locals.intermediates.push(ty)
    }

    /// A new constant table holding `tensor`.
    pub(crate) fn table(&mut self, tensor: HostTensor) -> TableId {
        let ty = TensorType::new(tensor.dtype(), tensor.shape().clone(), Layout::RowMajor);
        let _typed = self.locals.tables.push(ty);
        self.tables.push(tensor)
    }

    /// A stage over `shape` that may read the intermediates and tables
    /// declared so far.
    pub(crate) fn stage(&self, name: ComputeName, shape: &Shape) -> StageBuilder<'g> {
        StageBuilder::new(self.graph, self.locals.clone(), name, shape)
    }

    pub(crate) fn push(&mut self, stage: Stage) {
        self.stages.push(stage);
    }

    pub(crate) fn finish(self) -> LoweredGroup {
        LoweredGroup::new(self.locals.intermediates, self.tables, self.stages)
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
    locals: Locals,
    compute: ComputeBuilder,
    inputs: IndexVec<TensorParamId, Binding>,
    outputs: Vec<Binding>,
    spatial: Vec<ExprId>,
    reduce_axes: usize,
}

impl<'g> StageBuilder<'g> {
    /// A stage named `name` with one spatial axis per dimension of `shape`,
    /// reading graph values and the group's `locals`.
    pub(crate) fn new(graph: &'g Graph, locals: Locals, name: ComputeName, shape: &Shape) -> Self {
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
        Self { graph, locals, compute, inputs, outputs, spatial, reduce_axes: 0 }
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
            Binding::Intermediate(id) => &self.locals.intermediates[id],
            Binding::Table(id) => &self.locals.tables[id],
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
        Operand { expr: TensorExpr::load(tensor, index), dtype: self.type_of(binding).dtype() }
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
        Ok(Stage::new(compute, self.inputs, self.outputs))
    }
}
