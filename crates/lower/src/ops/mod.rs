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

use crate::error::LowerError;
use crate::stage::{Operand, Stage, StageBuilder};
use ptxc_core::Idx;
use ptxc_graph::{Graph, Node, NodeId, Op, ValueId};
use ptxc_loop::{ComputeName, TensorExpr};

/// One node to lower, with the outputs its group must compute.
pub(crate) struct NodeContext<'g> {
    graph: &'g Graph,
    id: NodeId,
    node: &'g Node,
    needed: &'g [ValueId],
}

/// The stages computing `cx`'s needed outputs.
pub(crate) fn lower_node(cx: &NodeContext<'_>) -> Result<Vec<Stage>, LowerError> {
    match cx.node.op() {
        Op::Unary(op) => elementwise::unary(cx, *op),
        Op::Binary(op) => elementwise::binary(cx, *op),
        Op::Variadic(op) => elementwise::variadic(cx, *op),
        Op::Where => elementwise::select(cx),
        Op::Cast(attrs) => elementwise::cast(cx, *attrs),
        Op::Activation(activation) => activation::activation(cx, *activation),
        Op::Clip => activation::clip(cx),
        Op::PRelu => activation::prelu(cx),
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

    /// One stage computing the node's single output elementwise from its
    /// operands, broadcast to the output; none if no one reads the output.
    fn elementwise(
        &self,
        expr: impl FnOnce(&[Operand]) -> Result<TensorExpr, LowerError>,
    ) -> Result<Vec<Stage>, LowerError> {
        let Some(&output) = self.node.outputs().iter().find(|value| self.needed.contains(value))
        else {
            return Ok(Vec::new());
        };
        let shape = self.graph.value(output).ty().shape();
        let mut stage = StageBuilder::new(self.graph, self.name(), shape);
        let operands: Vec<Operand> =
            self.node.operands().iter().map(|&value| stage.broadcast(value)).collect();
        stage.output(output, expr(&operands)?);
        let stage =
            stage.finish().map_err(|source| LowerError::Compute { node: self.id, source })?;
        Ok(vec![stage])
    }

    /// The node's `N` operands, or an error if it has another number.
    fn exact<'o, const N: usize>(
        &self,
        operands: &'o [Operand],
    ) -> Result<&'o [Operand; N], LowerError> {
        operands.try_into().map_err(|_| LowerError::Operands(self.id))
    }
}
