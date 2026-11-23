//! Per-family lowering, mirroring `ptxc_graph::ops`.

mod activation;
mod attention;
mod conv;
mod elementwise;
mod gather;
mod matmul;
mod norm;
mod pool;
mod quant;
mod reduce;
mod shape;
mod window;

use crate::builder::{GroupBuilder, Locals, Operand, StageBuilder};
use crate::error::LowerError;
use crate::stage::{Binding, LoweredGroup, Stage};
use ptxc_core::{DType, Dim, Idx, IndexVec};
use ptxc_graph::{Graph, Node, NodeId, Op, ValueId};
use ptxc_loop::{ComputeError, ComputeName, TensorExpr};

/// One node to lower, with the outputs its group must compute.
pub(crate) struct NodeContext<'g> {
    graph: &'g Graph,
    id: NodeId,
    node: &'g Node,
    needed: &'g [ValueId],
}

/// The stages computing `cx`'s needed outputs.
pub(crate) fn lower_node(cx: &NodeContext<'_>) -> Result<LoweredGroup, LowerError> {
    match cx.node.op() {
        Op::Unary(op) => elementwise::unary(cx, *op),
        Op::Binary(op) => elementwise::binary(cx, *op),
        Op::Variadic(op) => elementwise::variadic(cx, *op),
        Op::Where => elementwise::select(cx),
        Op::Cast(attrs) => elementwise::cast(cx, *attrs),
        Op::Activation(activation) => activation::activation(cx, *activation),
        Op::Clip => activation::clip(cx),
        Op::PRelu => activation::prelu(cx),
        Op::Reduce(attrs) => reduce::reduce(cx, attrs),
        Op::ArgReduce(attrs) => reduce::arg_reduce(cx, *attrs),
        Op::CumSum(attrs) => reduce::cum_sum(cx, *attrs),
        Op::Softmax(attrs) => norm::softmax(cx, *attrs),
        Op::LayerNorm(attrs) => norm::layer_norm(cx, *attrs),
        Op::BatchNorm(attrs) => norm::batch_norm(cx, *attrs),
        Op::InstanceNorm(attrs) => norm::instance_norm(cx, *attrs),
        Op::Reshape(target) => shape::reshape(cx, target),
        Op::Expand(_) => shape::expand(cx),
        Op::Transpose(attrs) => shape::transpose(cx, attrs),
        Op::Tile(_) => shape::tile(cx),
        Op::Concat(axis) => shape::concat(cx, *axis),
        Op::Slice(attrs) => shape::slice(cx, attrs),
        Op::Split(attrs) => shape::split(cx, attrs),
        Op::Pad(attrs) => shape::pad(cx, attrs),
        Op::Trilu(attrs) => shape::trilu(cx, *attrs),
        Op::Gather(axis) => gather::gather(cx, *axis),
        Op::GatherElements(axis) => gather::gather_elements(cx, *axis),
        Op::GatherNd(batch) => gather::gather_nd(cx, *batch),
        Op::MatMul => matmul::matmul(cx),
        Op::Gemm(attrs) => matmul::gemm(cx, *attrs),
        Op::Conv(attrs) => conv::conv(cx, attrs),
        Op::Pool(attrs) => pool::pool(cx, attrs),
        Op::GlobalPool(kind) => pool::global_pool(cx, *kind),
        Op::Quantize(attrs) => quant::quantize(cx, attrs),
        Op::Dequantize(attrs) => quant::dequantize(cx, attrs),
        Op::QMatMul(attrs) => quant::qmatmul(cx, attrs),
        Op::QConv(attrs) => quant::qconv(cx, attrs),
        other => Err(LowerError::Unsupported { op: other.mnemonic() }),
    }
}

impl<'g> NodeContext<'g> {
    pub(crate) const fn new(
        graph: &'g Graph,
        id: NodeId,
        node: &'g Node,
        needed: &'g [ValueId],
    ) -> Self {
        Self { graph, id, node, needed }
    }

    /// The name of the node's stage, from its op and id: `add_3`.
    fn name(&self) -> ComputeName {
        ComputeName::new(format!("{}_{}", self.node.op().mnemonic(), self.id.index()))
    }

    /// The name of one of the node's several stages: `reduce_log_sum_exp_3_max`.
    fn named(&self, part: &str) -> ComputeName {
        ComputeName::new(format!("{}_{}_{part}", self.node.op().mnemonic(), self.id.index()))
    }

    /// The node's single output, if its group must compute it.
    fn needed_output(&self) -> Option<ValueId> {
        self.node.outputs().iter().copied().find(|value| self.needed.contains(value))
    }

    /// The node's operands, in order.
    fn operands(&self) -> Vec<ValueId> {
        self.node.operands().iter().copied().collect()
    }

    /// The node's `N` operands, or an error if it has another number.
    fn values<const N: usize>(&self) -> Result<[ValueId; N], LowerError> {
        self.operands().try_into().map_err(|_| LowerError::Operands(self.id))
    }

    /// `dim` as an index-arithmetic constant; past `i64::MAX` it is the
    /// error `ComputeBuilder` would give.
    fn signed(&self, dim: Dim) -> Result<i64, LowerError> {
        i64::try_from(dim.get()).map_err(|_| self.ill_formed(ComputeError::ExtentTooLarge(dim)))
    }

    /// The error for this node's op on `dtype`, which has no lowering yet.
    fn unsupported(&self, dtype: DType) -> LowerError {
        LowerError::UnsupportedDType { op: self.node.op().mnemonic(), dtype }
    }

    /// A rejected definition as the error lowering this node gives.
    const fn ill_formed(&self, source: ComputeError) -> LowerError {
        LowerError::Compute { node: self.id, source }
    }

    /// The stage `stage` builds, if its definition is well-formed.
    fn finish(&self, stage: StageBuilder<'_>) -> Result<Stage, LowerError> {
        stage.finish().map_err(|source| self.ill_formed(source))
    }

    /// An empty group for this node's stages.
    const fn group(&self) -> GroupBuilder<'g> {
        GroupBuilder::new(self.graph)
    }

    /// One stage computing the node's single output as the expression
    /// `build` makes, over spatial axes of the output's shape; no stage if
    /// no one reads the output.
    fn one_stage(
        &self,
        build: impl FnOnce(&mut StageBuilder<'_>) -> Result<TensorExpr, LowerError>,
    ) -> Result<LoweredGroup, LowerError> {
        let Some(output) = self.needed_output() else { return Ok(LoweredGroup::default()) };
        let shape = self.graph.value(output).ty().shape();
        let mut stage = StageBuilder::new(self.graph, Locals::empty(), self.name(), shape);
        let expr = build(&mut stage)?;
        stage.output(Binding::Value(output), expr);
        Ok(LoweredGroup::new(IndexVec::new(), IndexVec::new(), vec![self.finish(stage)?]))
    }

    /// One stage computing the node's output elementwise from its operands,
    /// each broadcast to the output.
    fn elementwise(
        &self,
        expr: impl FnOnce(&[Operand]) -> Result<TensorExpr, LowerError>,
    ) -> Result<LoweredGroup, LowerError> {
        self.one_stage(|stage| {
            let operands: Vec<Operand> =
                self.operands().into_iter().map(|value| stage.broadcast(value)).collect();
            expr(&operands)
        })
    }

    /// The node's `N` operands, or an error if it has another number.
    fn exact<'o, const N: usize>(
        &self,
        operands: &'o [Operand],
    ) -> Result<&'o [Operand; N], LowerError> {
        operands.try_into().map_err(|_| LowerError::Operands(self.id))
    }
}
