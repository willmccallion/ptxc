//! Dead-code elimination.

use crate::pass::{Pass, PassError, PassReport};
use ptxc_graph::{Graph, NodeId, ValueDef};
use std::collections::HashSet;

/// Removes the nodes no graph output depends on and the constants nothing
/// reads, as XLA's `HloDCE`. A node with several outputs stays while any
/// of them is read.
#[derive(Clone, Copy, Debug)]
pub struct Dce;

impl Pass for Dce {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&self, graph: &mut Graph) -> Result<PassReport, PassError> {
        let live = live_nodes(graph);
        let mut dead = graph.topological_order()?;
        dead.retain(|node| !live.contains(node));
        let changed = !dead.is_empty() || has_unread_constant(graph);
        let mut rewriter = graph.rewriter();
        for &node in dead.iter().rev() {
            rewriter.erase_node(node)?;
        }
        rewriter.compact();
        Ok(PassReport { changed })
    }
}

/// The nodes some graph output depends on.
fn live_nodes(graph: &Graph) -> HashSet<NodeId> {
    let mut live = HashSet::new();
    let mut pending: Vec<NodeId> =
        graph.outputs().iter().filter_map(|&value| graph.producer(value)).collect();
    while let Some(node) = pending.pop() {
        if !live.insert(node) {
            continue;
        }
        let Some(contents) = graph.node(node) else { continue };
        pending.extend(contents.operands().iter().filter_map(|&value| graph.producer(value)));
    }
    live
}

/// Whether some constant of `graph` has no reader.
fn has_unread_constant(graph: &Graph) -> bool {
    graph.values().any(|(id, value)| {
        matches!(value.def(), ValueDef::Constant(_)) && graph.uses(id).is_empty()
    })
}
