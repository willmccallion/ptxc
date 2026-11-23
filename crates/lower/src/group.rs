//! Lowers each group of a partitioned graph to its stages.

use crate::error::LowerError;
use crate::ops::{NodeContext, lower_node};
use crate::stage::Stage;
use ptxc_core::IndexVec;
use ptxc_graph::{FusedGroup, GroupId, Partitioned};

/// Every group's stages, in the order they run; a stage reads only graph
/// values computed by earlier groups or stages.
pub fn lower(partitioned: &Partitioned) -> Result<IndexVec<GroupId, Vec<Stage>>, LowerError> {
    partitioned
        .groups()
        .iter_enumerated()
        .map(|(id, group)| lower_group(partitioned, id, group))
        .collect()
}

fn lower_group(
    partitioned: &Partitioned,
    id: GroupId,
    group: &FusedGroup,
) -> Result<Vec<Stage>, LowerError> {
    let graph = partitioned.graph();
    let nodes: Vec<_> =
        group.nodes().iter().filter_map(|&node| Some((node, graph.node(node)?))).collect();
    let [(node_id, node)] = nodes[..] else { return Err(LowerError::Fused(id)) };
    lower_node(&NodeContext::new(graph, node_id, node, group.outputs()))
}
