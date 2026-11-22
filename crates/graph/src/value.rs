//! `Value`, `ValueId` and `ValueDef`, the single definition of each SSA value,
//! and `Use`, one place a value is read.

use crate::node::{NodeId, NodeOutputIndex, OperandIndex};
use crate::tensor_type::TensorType;
use ptxc_core::define_index;

define_index!(
    /// An SSA value in a graph.
    pub ValueId
);

define_index!(
    /// A constant tensor in a graph.
    pub ConstantId
);

define_index!(
    /// A position in a graph's inputs.
    pub GraphInputIndex
);

define_index!(
    /// A position in a graph's outputs.
    pub GraphOutputIndex
);

/// A value's name from the source model, kept for diagnostics and the text
/// format.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ValueName(String);

impl ValueName {
    /// Wraps a name.
    pub const fn new(name: String) -> Self {
        Self(name)
    }

    /// The name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Where a value is defined. Each value has exactly one definition.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ValueDef {
    /// A graph input.
    Input(GraphInputIndex),
    /// A constant tensor.
    Constant(ConstantId),
    /// An output of a node.
    Output {
        /// The defining node.
        node: NodeId,
        /// Which of its outputs.
        index: NodeOutputIndex,
    },
}

/// An SSA value: its type, its definition and an optional name.
#[derive(Clone, Debug)]
pub struct Value {
    pub(crate) ty: TensorType,
    pub(crate) def: ValueDef,
    pub(crate) name: Option<ValueName>,
}

impl Value {
    /// The value's type.
    pub const fn ty(&self) -> &TensorType {
        &self.ty
    }

    /// Where the value is defined.
    pub const fn def(&self) -> ValueDef {
        self.def
    }

    /// The name from the source model, if it had one.
    pub const fn name(&self) -> Option<&ValueName> {
        self.name.as_ref()
    }
}

/// One place a value is read.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Use {
    /// An operand of a node.
    Operand {
        /// The reading node.
        node: NodeId,
        /// Which of its operands.
        index: OperandIndex,
    },
    /// A graph output.
    GraphOutput(GraphOutputIndex),
}
