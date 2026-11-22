//! `Graph`: node, value and constant arenas plus the graph's inputs and outputs.

use crate::node::{Node, NodeId, NodeOutputIndex, NodeSlot, OperandIndex};
use crate::ops::{Op, TypeError, TypeRule};
use crate::tensor_type::{Layout, TensorType};
use crate::value::{ConstantId, GraphInputIndex, GraphOutputIndex, Use, Value, ValueDef, ValueId};
use ptxc_core::{HostTensor, IndexVec};
use thiserror::Error;

/// A node that cannot be added.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum NodeError {
    /// An operand that is not a live value of this graph.
    #[error("operand {0:?} is not a live value of this graph")]
    NotLive(ValueId),
    /// Operands the op's type rule rejects.
    #[error("{op:?}: {source}")]
    Type {
        /// The op.
        op: Op,
        /// Why its type rule rejected the operands.
        source: TypeError,
    },
}

/// An SSA dataflow graph in typed arenas, in the style of XLA's HLO module
/// (D2). Built by `GraphBuilder` and changed only through `Rewriter`, which
/// keep the use lists in step with the nodes.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub(crate) nodes: IndexVec<NodeId, NodeSlot>,
    pub(crate) values: IndexVec<ValueId, Value>,
    pub(crate) constants: IndexVec<ConstantId, HostTensor>,
    pub(crate) uses: IndexVec<ValueId, Vec<Use>>,
    pub(crate) inputs: IndexVec<GraphInputIndex, ValueId>,
    pub(crate) outputs: IndexVec<GraphOutputIndex, ValueId>,
}

impl Graph {
    /// The node `id`, or `None` if it has been erased.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id).and_then(NodeSlot::live)
    }

    /// The live nodes in id order, which is not in general a topological
    /// order once a graph has been rewritten.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter_enumerated().filter_map(|(id, slot)| Some((id, slot.live()?)))
    }

    /// The value `id`, which must come from this graph.
    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id]
    }

    /// The constant tensor `id`, which must come from this graph.
    pub fn constant(&self, id: ConstantId) -> &HostTensor {
        &self.constants[id]
    }

    /// Every place `value`, a value of this graph, is read.
    pub fn uses(&self, value: ValueId) -> &[Use] {
        &self.uses[value]
    }

    /// The graph inputs, in order.
    pub const fn inputs(&self) -> &IndexVec<GraphInputIndex, ValueId> {
        &self.inputs
    }

    /// The graph outputs, in order.
    pub const fn outputs(&self) -> &IndexVec<GraphOutputIndex, ValueId> {
        &self.outputs
    }

    /// The node defining `value`, if a node defines it.
    pub fn producer(&self, value: ValueId) -> Option<NodeId> {
        match self.values.get(value)?.def {
            ValueDef::Output { node, .. } => Some(node),
            ValueDef::Input(_) | ValueDef::Constant(_) => None,
        }
    }

    /// Whether `id` is a value of this graph whose definition still exists.
    pub fn is_live(&self, id: ValueId) -> bool {
        self.values.get(id).is_some_and(|value| match value.def {
            ValueDef::Output { node, .. } => self.node(node).is_some(),
            ValueDef::Input(_) | ValueDef::Constant(_) => true,
        })
    }

    pub(crate) fn push_input(&mut self, ty: TensorType) -> ValueId {
        let index = self.inputs.next_id();
        let value = self.push_value(ty, ValueDef::Input(index));
        let _: GraphInputIndex = self.inputs.push(value);
        value
    }

    pub(crate) fn push_constant(&mut self, tensor: HostTensor) -> ValueId {
        let ty = TensorType::new(tensor.dtype(), tensor.shape().clone(), Layout::RowMajor);
        let constant = self.constants.push(tensor);
        self.push_value(ty, ValueDef::Constant(constant))
    }

    /// Adds a node, inferring its output types from its op's type rule.
    pub(crate) fn push_node(
        &mut self,
        op: Op,
        operands: Vec<ValueId>,
    ) -> Result<Vec<ValueId>, NodeError> {
        let output_types = self.infer(&op, &operands)?;
        let node = self.nodes.next_id();
        let mut outputs: IndexVec<NodeOutputIndex, ValueId> = IndexVec::new();
        for ty in output_types {
            let value = self.push_value(ty, ValueDef::Output { node, index: outputs.next_id() });
            let _: NodeOutputIndex = outputs.push(value);
        }
        let operands: IndexVec<OperandIndex, ValueId> = operands.into_iter().collect();
        for (index, &operand) in operands.iter_enumerated() {
            self.uses[operand].push(Use::Operand { node, index });
        }
        let output_ids = outputs.iter().copied().collect();
        let _: NodeId = self.nodes.push(NodeSlot::Live(Node { op, operands, outputs }));
        Ok(output_ids)
    }

    fn infer(&self, op: &Op, operands: &[ValueId]) -> Result<Vec<TensorType>, NodeError> {
        let types = operands
            .iter()
            .map(|&id| {
                if self.is_live(id) { Ok(&self.values[id].ty) } else { Err(NodeError::NotLive(id)) }
            })
            .collect::<Result<Vec<_>, _>>()?;
        op.infer(&types).map_err(|source| NodeError::Type { op: op.clone(), source })
    }

    fn push_value(&mut self, ty: TensorType, def: ValueDef) -> ValueId {
        let _: ValueId = self.uses.push(Vec::new());
        self.values.push(Value { ty, def, name: None })
    }
}
