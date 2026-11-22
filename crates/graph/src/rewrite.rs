//! `Rewriter`: the only API that mutates a graph, keeping use lists consistent.

use crate::graph::{Graph, NodeError};
use crate::node::{NodeId, NodeSlot};
use crate::ops::Op;
use crate::tensor_type::TensorType;
use crate::value::{Use, ValueId};
use ptxc_core::HostTensor;
use thiserror::Error;

/// A rewrite that would break the graph's invariants.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum RewriteError {
    /// The value is not a live value of this graph.
    #[error(transparent)]
    Node(#[from] NodeError),
    /// A replacement whose type differs from the value it replaces.
    #[error("cannot replace {old:?} ({old_type:?}) with {new:?} ({new_type:?})")]
    TypeMismatch {
        /// The value being replaced.
        old: ValueId,
        /// Its type.
        old_type: TensorType,
        /// The replacement.
        new: ValueId,
        /// Its type.
        new_type: TensorType,
    },
    /// The node is erased or not in this graph.
    #[error("node {0:?} is not a live node of this graph")]
    NodeNotLive(NodeId),
    /// Erasing a node whose output is still read.
    #[error("cannot erase {node:?}: its output {value:?} is still used")]
    OutputStillUsed {
        /// The node.
        node: NodeId,
        /// The output that still has uses.
        value: ValueId,
    },
}

/// Mutates a graph through operations that each keep the use lists exact,
/// as MLIR's `RewriterBase`.
///
/// Acyclicity is the caller's to keep: `replace_all_uses` with a value
/// computed from `old`'s users makes a cycle, which the verifier reports.
#[derive(Debug)]
pub struct Rewriter<'g> {
    graph: &'g mut Graph,
}

impl Graph {
    /// A rewriter over this graph.
    pub const fn rewriter(&mut self) -> Rewriter<'_> {
        Rewriter { graph: self }
    }
}

impl Rewriter<'_> {
    /// The graph as it stands.
    pub const fn graph(&self) -> &Graph {
        self.graph
    }

    /// Adds a node reading `operands` and returns the values it defines.
    pub fn add_node(&mut self, op: Op, operands: Vec<ValueId>) -> Result<Vec<ValueId>, NodeError> {
        self.graph.push_node(op, operands)
    }

    /// Adds a constant tensor; its value has the tensor's type, row-major.
    pub fn add_constant(&mut self, tensor: HostTensor) -> ValueId {
        self.graph.push_constant(tensor)
    }

    /// Makes every reader of `old`, graph outputs included, read `new`.
    pub fn replace_all_uses(&mut self, old: ValueId, new: ValueId) -> Result<(), RewriteError> {
        for value in [old, new] {
            if !self.graph.is_live(value) {
                return Err(NodeError::NotLive(value).into());
            }
        }
        let (old_type, new_type) = (&self.graph.values[old].ty, &self.graph.values[new].ty);
        if old_type != new_type {
            let (old_type, new_type) = (old_type.clone(), new_type.clone());
            return Err(RewriteError::TypeMismatch { old, old_type, new, new_type });
        }
        if old == new {
            return Ok(());
        }
        let moved = std::mem::take(&mut self.graph.uses[old]);
        for &reader in &moved {
            self.redirect(reader, new);
        }
        self.graph.uses[new].extend(moved);
        Ok(())
    }

    /// Erases a node none of whose outputs is still used.
    pub fn erase_node(&mut self, id: NodeId) -> Result<(), RewriteError> {
        let Some(node) = self.graph.node(id) else {
            return Err(RewriteError::NodeNotLive(id));
        };
        if let Some(&value) = node.outputs.iter().find(|&&value| !self.graph.uses[value].is_empty())
        {
            return Err(RewriteError::OutputStillUsed { node: id, value });
        }
        let operands: Vec<_> = node.operands.iter_enumerated().map(|(i, &v)| (i, v)).collect();
        for (index, operand) in operands {
            self.graph.uses[operand].retain(|&reader| reader != Use::Operand { node: id, index });
        }
        self.graph.nodes[id] = NodeSlot::Erased;
        Ok(())
    }

    /// Removes erased nodes and the values they defined, renumbering every
    /// node and value id. Ids held from before are invalid afterwards.
    pub fn compact(self) {
        crate::compact::compact(self.graph);
    }

    fn redirect(&mut self, reader: Use, new: ValueId) {
        match reader {
            Use::Operand { node, index } => {
                if let NodeSlot::Live(node) = &mut self.graph.nodes[node] {
                    node.operands[index] = new;
                }
            }
            Use::GraphOutput(index) => self.graph.outputs[index] = new,
        }
    }
}
