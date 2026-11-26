//! Greedy producer-consumer fusion.

use super::partitioner::Partitioner;
use super::policy::{Role, role, takes_into_reads};
use ptxc_graph::{Graph, NodeId, PartitionError, Partitioned, Use, ValueId};
use std::collections::{HashMap, HashSet};

/// XLA's producer-consumer fusion, greedily (D82): each group, taken from
/// the last, absorbs a producer group once every reader of the producer's
/// outputs is in it.
///
/// A producer that reduces joins only consumers that read it elementwise
/// at each of its elements, so its reduction becomes their epilogue's; one
/// that does not is inlined into its readers' loads where `policy` allows.
/// A group holds at most one reduction or product.
#[derive(Clone, Copy, Debug)]
pub struct GreedyPartitioner;

impl Partitioner for GreedyPartitioner {
    fn partition(&self, graph: Graph) -> Result<Partitioned, PartitionError> {
        let groups = Groups::singletons(&graph)?.fused();
        Partitioned::from_groups(graph, &groups)
    }
}

/// Groups of nodes as they merge: each node's group, and each group's
/// nodes, empty once merged away.
struct Groups<'g> {
    graph: &'g Graph,
    order: Vec<NodeId>,
    group_of: HashMap<NodeId, usize>,
    members: Vec<Vec<NodeId>>,
}

impl<'g> Groups<'g> {
    /// Each node of `graph` its own group.
    fn singletons(graph: &'g Graph) -> Result<Self, PartitionError> {
        let order = graph.topological_order()?;
        let group_of = order.iter().enumerate().map(|(group, &node)| (node, group)).collect();
        let members = order.iter().map(|&node| vec![node]).collect();
        Ok(Self { graph, order, group_of, members })
    }

    /// The groups after each, from the last, has absorbed every producer
    /// group it can.
    fn fused(mut self) -> Vec<Vec<NodeId>> {
        for &node in self.order.clone().iter().rev() {
            let group = self.group_of[&node];
            while let Some(producer) = self.absorbable_producer(group) {
                self.merge(producer, group);
            }
        }
        self.members.into_iter().filter(|members| !members.is_empty()).collect()
    }

    /// A group computing a value `group` reads that `group` may absorb.
    fn absorbable_producer(&self, group: usize) -> Option<usize> {
        let producers: HashSet<usize> = self.members[group]
            .iter()
            .filter_map(|&node| self.graph.node(node))
            .flat_map(|node| node.operands().iter())
            .filter_map(|&value| Some(self.group_of[&self.graph.producer(value)?]))
            .filter(|&producer| producer != group)
            .collect();
        let mut producers: Vec<usize> = producers.into_iter().collect();
        producers.sort_unstable();
        producers.into_iter().find(|&producer| self.absorbs(group, producer))
    }

    /// Whether `group` may absorb `producer`: every reader of what
    /// `producer` computes is in `group`, and `policy` lets those readers
    /// take it.
    fn absorbs(&self, group: usize, producer: usize) -> bool {
        let roles: Vec<Role> = self.roles(producer).collect();
        if roles.contains(&Role::Apart) {
            return false;
        }
        let Some(readers) = self.readers(producer, group) else { return false };
        let reduces = roles.iter().any(|role| matches!(role, Role::Anchor(_)));
        if !reduces {
            return readers.iter().all(|&(reader, _)| self.takes_into_reads(reader, &roles));
        }
        let anchored = self.roles(group).any(|role| matches!(role, Role::Anchor(_)));
        !anchored && readers.iter().all(|&(reader, value)| self.reads_elementwise(reader, value))
    }

    /// Each node of `group` reading a value `producer` computes, with the
    /// value; `None` if any other node reads one, or one is a graph output.
    ///
    /// Groups are walked from the last, so a producer is still a single
    /// node when a group that reads it is reached: none of its values is
    /// read within it.
    fn readers(&self, producer: usize, group: usize) -> Option<Vec<(NodeId, ValueId)>> {
        let mut readers = Vec::new();
        for &node in &self.members[producer] {
            let outputs =
                self.graph.node(node).map(|node| node.outputs().iter()).into_iter().flatten();
            for &value in outputs {
                for reader in self.graph.uses(value) {
                    match *reader {
                        Use::Operand { node: by, .. } if self.group_of[&by] == group => {
                            readers.push((by, value));
                        }
                        _ => return None,
                    }
                }
            }
        }
        Some(readers)
    }

    /// Whether `reader` may have a group of ops of `roles` inlined into its
    /// loads.
    fn takes_into_reads(&self, reader: NodeId, roles: &[Role]) -> bool {
        self.graph.node(reader).is_some_and(|node| takes_into_reads(node.op(), roles))
    }

    /// Whether `reader` is elementwise and reads `value` at each of its own
    /// elements: its output has `value`'s shape.
    fn reads_elementwise(&self, reader: NodeId, value: ValueId) -> bool {
        let Some(node) = self.graph.node(reader) else { return false };
        let shape = |value: ValueId| self.graph.value(value).ty().shape();
        role(node.op()) == Role::Elementwise
            && node.outputs().iter().all(|&output| shape(output) == shape(value))
    }

    /// The roles of `group`'s ops.
    fn roles(&self, group: usize) -> impl Iterator<Item = Role> + '_ {
        self.members[group].iter().filter_map(|&node| Some(role(self.graph.node(node)?.op())))
    }

    /// Moves `producer`'s nodes into `group`.
    fn merge(&mut self, producer: usize, group: usize) {
        let moved = std::mem::take(&mut self.members[producer]);
        for &node in &moved {
            let _old = self.group_of.insert(node, group);
        }
        self.members[group].extend(moved);
    }
}
