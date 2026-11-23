//! `Stage`: a compute definition bound to the graph values it reads and
//! writes, and the builder lowering uses to make one.

use ptxc_core::{DType, Dim, IndexVec, Shape};
use ptxc_graph::{Graph, ValueId};
use ptxc_loop::{
    ComputeBuilder, ComputeDef, ComputeError, ComputeName, ExprId, TensorExpr, TensorName,
    TensorParamId, VarName,
};
use std::sync::Arc;

/// One compute definition and the graph values bound to its tensors. Each
/// stage is one kernel in M2.
#[derive(Clone, Debug)]
pub struct Stage {
    compute: Arc<ComputeDef>,
    inputs: IndexVec<TensorParamId, ValueId>,
    outputs: Vec<ValueId>,
}

impl Stage {
    /// What the stage computes.
    pub const fn compute(&self) -> &Arc<ComputeDef> {
        &self.compute
    }

    /// The value bound to each of the definition's input tensors.
    pub const fn inputs(&self) -> &IndexVec<TensorParamId, ValueId> {
        &self.inputs
    }

    /// The value each of the definition's outputs computes, in order.
    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }
}

/// An operand loaded at the output element it broadcasts to.
#[derive(Clone, Debug)]
pub(crate) struct Operand {
    pub(crate) expr: TensorExpr,
    pub(crate) dtype: DType,
}

/// Builds a stage whose spatial axes are an output shape's dimensions.
pub(crate) struct StageBuilder<'g> {
    graph: &'g Graph,
    compute: ComputeBuilder,
    inputs: IndexVec<TensorParamId, ValueId>,
    outputs: Vec<ValueId>,
    axes: Vec<ExprId>,
}

impl<'g> StageBuilder<'g> {
    /// A stage named `name` with one spatial axis per dimension of `shape`.
    pub(crate) fn new(graph: &'g Graph, name: ComputeName, shape: &Shape) -> Self {
        let mut compute = ComputeBuilder::new(name);
        let axes = shape
            .dims()
            .iter()
            .enumerate()
            .map(|(position, &extent)| axis(&mut compute, position, extent))
            .collect();
        Self { graph, compute, inputs: IndexVec::new(), outputs: Vec::new(), axes }
    }

    /// `value`'s tensor, added as an input the first time it is read.
    fn tensor(&mut self, value: ValueId) -> TensorParamId {
        if let Some((tensor, _)) = self.inputs.iter_enumerated().find(|&(_, &bound)| bound == value)
        {
            return tensor;
        }
        let ty = self.graph.value(value).ty();
        let name = TensorName::new(format!("in{}", self.inputs.iter().len()));
        let tensor = self.compute.input(name, ty.dtype(), ty.shape().clone());
        let _bound = self.inputs.push(value);
        tensor
    }

    /// `value` read at the output element, with ONNX's multidirectional
    /// broadcasting: trailing axes align, and an axis of extent 1 is read
    /// at 0.
    pub(crate) fn broadcast(&mut self, value: ValueId) -> Operand {
        let tensor = self.tensor(value);
        let ty = self.graph.value(value).ty();
        let zero = self.compute.exprs().constant(0);
        let mut index: Vec<ExprId> = ty
            .shape()
            .dims()
            .iter()
            .rev()
            .zip(self.axes.iter().rev())
            .map(|(dim, &axis)| if dim.get() == 1 { zero } else { axis })
            .collect();
        index.reverse();
        Operand { expr: TensorExpr::Load { tensor, index }, dtype: ty.dtype() }
    }

    /// Adds an output computing `value` as `expr`.
    pub(crate) fn output(&mut self, value: ValueId, expr: TensorExpr) {
        let name = TensorName::new(format!("out{}", self.outputs.len()));
        self.compute.output(name, expr);
        self.outputs.push(value);
    }

    /// The stage, if its definition is well-formed.
    pub(crate) fn finish(self) -> Result<Stage, ComputeError> {
        let compute = Arc::new(self.compute.finish()?);
        Ok(Stage { compute, inputs: self.inputs, outputs: self.outputs })
    }
}

/// Spatial axis `position`, named `i<position>`.
fn axis(compute: &mut ComputeBuilder, position: usize, extent: Dim) -> ExprId {
    compute.spatial(VarName::new(format!("i{position}")), extent).1
}
