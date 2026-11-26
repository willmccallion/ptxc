//! Renumbering a graph's arenas to drop erased nodes and the values they
//! defined, and constants nothing reads (D2).

use crate::graph::Graph;
use crate::node::{Node, NodeId, NodeSlot};
use crate::value::{ConstantId, Use, Value, ValueDef, ValueId};
use ptxc_core::{Idx, IndexVec};
use std::mem::take;

/// Drops tombstones and constants nothing reads, and renumbers every node,
/// value and constant id densely, keeping the relative order of what
/// remains.
pub(crate) fn compact(graph: &mut Graph) {
    let value_kept: Vec<bool> = graph.values.ids().map(|id| is_kept(graph, id)).collect();
    let mut constant_kept: IndexVec<ConstantId, bool> =
        graph.constants.iter().map(|_| false).collect();
    for (value, kept) in graph.values.iter().zip(&value_kept) {
        if let (ValueDef::Constant(constant), true) = (value.def, kept) {
            constant_kept[constant] = true;
        }
    }
    let remap = Remap {
        nodes: renumber(graph.nodes.iter().map(|slot| slot.live().is_some())),
        values: renumber(value_kept.iter().copied()),
        constants: renumber(constant_kept.iter().copied()),
    };
    graph.constants = take(&mut graph.constants)
        .into_iter()
        .zip(constant_kept.iter())
        .filter_map(|(tensor, &kept)| kept.then_some(tensor))
        .collect();

    graph.nodes = take(&mut graph.nodes)
        .into_iter()
        .filter_map(|slot| match slot {
            NodeSlot::Live(node) => Some(NodeSlot::Live(remap.node(node))),
            NodeSlot::Erased => None,
        })
        .collect();

    let old_values = take(&mut graph.values).into_iter().zip(take(&mut graph.uses));
    for ((value, uses), kept) in old_values.zip(value_kept) {
        if kept {
            let _: ValueId = graph.values.push(remap.value(value));
            let _: ValueId = graph.uses.push(remap.uses(uses));
        }
    }

    graph.inputs = take(&mut graph.inputs).into_iter().map(|value| remap.values[value]).collect();
    graph.outputs = take(&mut graph.outputs).into_iter().map(|value| remap.values[value]).collect();
}

/// Whether compaction keeps `value`: it is live, and if a constant, read.
fn is_kept(graph: &Graph, value: ValueId) -> bool {
    let unread_constant =
        matches!(graph.values[value].def, ValueDef::Constant(_)) && graph.uses[value].is_empty();
    graph.is_live(value) && !unread_constant
}

/// Old id to new id, for each arena.
struct Remap {
    nodes: IndexVec<NodeId, NodeId>,
    values: IndexVec<ValueId, ValueId>,
    constants: IndexVec<ConstantId, ConstantId>,
}

impl Remap {
    fn node(&self, node: Node) -> Node {
        Node {
            op: node.op,
            operands: node.operands.into_iter().map(|value| self.values[value]).collect(),
            outputs: node.outputs.into_iter().map(|value| self.values[value]).collect(),
        }
    }

    fn value(&self, value: Value) -> Value {
        let def = match value.def {
            ValueDef::Output { node, index } => ValueDef::Output { node: self.nodes[node], index },
            ValueDef::Constant(constant) => ValueDef::Constant(self.constants[constant]),
            def @ ValueDef::Input(_) => def,
        };
        Value { def, ..value }
    }

    fn uses(&self, uses: Vec<Use>) -> Vec<Use> {
        uses.into_iter()
            .map(|reader| match reader {
                Use::Operand { node, index } => Use::Operand { node: self.nodes[node], index },
                Use::GraphOutput(_) => reader,
            })
            .collect()
    }
}

/// Each old id's new id: its position among the kept entries. A dropped
/// entry maps to its successor's id, which nothing reads.
fn renumber<Old: Idx, New: Idx>(kept: impl Iterator<Item = bool>) -> IndexVec<Old, New> {
    let mut next = 0;
    kept.map(|keep| {
        let id = New::new(next);
        next += usize::from(keep);
        id
    })
    .collect()
}
