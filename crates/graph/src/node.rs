//! `Node` and `NodeId`.

use crate::ops::Op;
use crate::value::ValueId;
use ptxc_core::{IndexVec, define_index};

define_index!(
    /// A node in a graph. Ids stay stable until `Rewriter::compact`.
    pub NodeId
);

define_index!(
    /// A position in a node's operands.
    pub OperandIndex
);

define_index!(
    /// A position in a node's outputs.
    pub NodeOutputIndex
);

/// One operation: its op, the values it reads and the values it defines.
#[derive(Clone, Debug)]
pub struct Node {
    pub(crate) op: Op,
    pub(crate) operands: IndexVec<OperandIndex, ValueId>,
    pub(crate) outputs: IndexVec<NodeOutputIndex, ValueId>,
}

impl Node {
    /// The operation.
    pub const fn op(&self) -> &Op {
        &self.op
    }

    /// The values read, in operand order.
    pub const fn operands(&self) -> &IndexVec<OperandIndex, ValueId> {
        &self.operands
    }

    /// The values defined, in output order.
    pub const fn outputs(&self) -> &IndexVec<NodeOutputIndex, ValueId> {
        &self.outputs
    }
}

/// A node arena slot. Erasing leaves a tombstone so other ids stay valid
/// until `compact` renumbers (D2).
#[derive(Clone, Debug)]
pub(crate) enum NodeSlot {
    Live(Node),
    Erased,
}

impl NodeSlot {
    pub(crate) const fn live(&self) -> Option<&Node> {
        match self {
            Self::Live(node) => Some(node),
            Self::Erased => None,
        }
    }
}
