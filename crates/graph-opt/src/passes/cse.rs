//! Common-subexpression elimination.

use crate::pass::{Pass, PassError, PassReport};
use ptxc_core::{DType, Shape};
use ptxc_graph::{Graph, NodeId, Op, Rewriter, ValueDef, ValueId};
use std::collections::HashMap;

/// Merges nodes with the same op, attributes and operands, and constants
/// with the same type and bytes, as XLA's `HloCSE`.
///
/// Every reader of the later moves to the earlier, and the later is
/// removed. Ops are deterministic (D2), so equal nodes compute equal values.
#[derive(Clone, Copy, Debug)]
pub struct Cse;

impl Pass for Cse {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run(&self, graph: &mut Graph) -> Result<PassReport, PassError> {
        let order = graph.topological_order()?;
        let mut rewriter = graph.rewriter();
        let constants = merge_constants(&mut rewriter)?;
        let nodes = merge_nodes(&mut rewriter, &order)?;
        rewriter.compact();
        Ok(PassReport { changed: constants || nodes })
    }
}

/// Moves the readers of each constant to the first constant with its type
/// and bytes; whether any moved.
fn merge_constants(rewriter: &mut Rewriter<'_>) -> Result<bool, PassError> {
    let graph = rewriter.graph();
    let mut first: HashMap<(DType, Shape, Vec<u8>), ValueId> = HashMap::new();
    let mut repeats = Vec::new();
    for (id, value) in graph.values() {
        let ValueDef::Constant(constant) = value.def() else { continue };
        let tensor = graph.constant(constant);
        let key = (tensor.dtype(), tensor.shape().clone(), tensor.data().to_le_bytes());
        if let Some(&earlier) = first.get(&key) {
            repeats.push((id, earlier));
        } else {
            let _new = first.insert(key, id);
        }
    }
    for &(repeat, earlier) in &repeats {
        rewriter.replace_all_uses(repeat, earlier)?;
    }
    Ok(!repeats.is_empty())
}

/// Moves the readers of each node in `order`, a topological order, to the
/// first node with its op and operands, and erases it; whether any did.
fn merge_nodes(rewriter: &mut Rewriter<'_>, order: &[NodeId]) -> Result<bool, PassError> {
    let mut first: HashMap<(Op, Vec<ValueId>), Vec<ValueId>> = HashMap::new();
    let mut merged = false;
    for &node in order {
        let Some(contents) = rewriter.graph().node(node) else { continue };
        let key = (contents.op().clone(), contents.operands().iter().copied().collect());
        let outputs: Vec<ValueId> = contents.outputs().iter().copied().collect();
        let Some(kept) = first.get(&key) else {
            let _new = first.insert(key, outputs);
            continue;
        };
        for (&repeat, &kept) in outputs.iter().zip(kept) {
            rewriter.replace_all_uses(repeat, kept)?;
        }
        rewriter.erase_node(node)?;
        merged = true;
    }
    Ok(merged)
}
