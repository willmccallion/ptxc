//! What a node's importer produces: graph nodes, constants, or an alias.

use ptxc_core::HostTensor;
use ptxc_graph::Op;

/// An operand of a lowered node.
#[derive(Debug)]
pub(crate) enum Operand {
    /// The node's input at this position.
    Input(usize),
    /// A constant the importer adds, such as a default `Clip` bound.
    Constant(HostTensor),
}

/// A node's import in canonical form (D32).
#[derive(Debug)]
pub(crate) enum Lowered {
    /// One graph node; its outputs are the node's outputs in order.
    Node { op: Op, operands: Vec<Operand> },
    /// Two graph nodes: `first` on the node's input 0, then `then` on
    /// `first`'s output followed by `rest` (a cropping `Pad` is a `Slice`
    /// then a `Pad`). `then`'s outputs are the node's outputs.
    Chain { first: Op, then: Op, rest: Vec<Operand> },
    /// Values known at import, one per output (`Shape`, `Constant`, ...).
    Constants(Vec<HostTensor>),
    /// The node's output is its input at this position (`Identity`).
    Alias(usize),
}

impl Lowered {
    /// A node reading the node's first `count` inputs in order.
    pub(crate) fn on_inputs(op: Op, count: usize) -> Self {
        Self::Node { op, operands: (0..count).map(Operand::Input).collect() }
    }
}
