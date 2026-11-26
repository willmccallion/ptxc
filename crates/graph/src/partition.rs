//! `FusedGroup` and `Partitioned`, a graph whose every node belongs to exactly one group.

use crate::graph::Graph;
use crate::node::{Node, NodeId};
use crate::ops::Op;
use crate::traverse::CycleError;
use crate::value::{Use, ValueId};
use ptxc_core::{IndexVec, define_index};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use thiserror::Error;

define_index!(
    /// A fused group in a partitioned graph.
    pub GroupId
);

/// What a group computes, which decides how it is scheduled and fused.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GroupKind {
    /// Each output element reads a fixed set of input elements: elementwise
    /// ops, data movement and gathers.
    Injective,
    /// Reduces along one or more axes: reductions, scans, normalisation,
    /// pooling.
    Reduction,
    /// A matrix product and the injective ops fused after it.
    MatMulEpilogue,
    /// A convolution and the injective ops fused after it.
    ConvEpilogue,
    /// A whole attention block.
    Attention,
    /// A reshape, which reorders nothing in a row-major tensor.
    View,
}

impl GroupKind {
    /// The kind of a group holding only `op`.
    pub const fn of(op: &Op) -> Self {
        match op {
            Op::Reshape(_) => Self::View,
            Op::MatMul | Op::Gemm(_) | Op::MatMulInteger(_) | Op::QMatMul(_) => {
                Self::MatMulEpilogue
            }
            Op::Conv(_) | Op::QConv(_) => Self::ConvEpilogue,
            Op::Attention(_) => Self::Attention,
            Op::Pool(_)
            | Op::GlobalPool(_)
            | Op::Reduce(_)
            | Op::ArgReduce(_)
            | Op::CumSum(_)
            | Op::Softmax(_)
            | Op::LayerNorm(_)
            | Op::InstanceNorm(_) => Self::Reduction,
            Op::Unary(_)
            | Op::Binary(_)
            | Op::Variadic(_)
            | Op::Where
            | Op::Cast(_)
            | Op::Activation(_)
            | Op::Clip
            | Op::PRelu
            | Op::BatchNorm(_)
            | Op::Expand(_)
            | Op::Transpose(_)
            | Op::Tile(_)
            | Op::Concat(_)
            | Op::Slice(_)
            | Op::Split(_)
            | Op::Pad(_)
            | Op::Trilu(_)
            | Op::Gather(_)
            | Op::GatherElements(_)
            | Op::GatherNd(_)
            | Op::Quantize(_)
            | Op::Dequantize(_)
            | Op::Resize(_) => Self::Injective,
        }
    }
}

/// Nodes that run as one unit, with the values crossing its boundary.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FusedGroup {
    nodes: Vec<NodeId>,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
    kind: GroupKind,
}

impl FusedGroup {
    /// The group's nodes, in topological order.
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// The values the group reads and does not compute, each listed once,
    /// in the order its nodes first read them.
    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    /// The values the group computes that are read outside it or are graph
    /// outputs, in the order its nodes define them.
    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    /// What the group computes.
    pub const fn kind(&self) -> GroupKind {
        self.kind
    }
}

/// A graph whose every live node belongs to exactly one group, an
/// invariant `Graph` cannot express, as XLA's fusion output.
///
/// Groups are numbered in an order where every group comes after the groups
/// computing its inputs.
#[derive(Clone, Debug)]
pub struct Partitioned {
    graph: Graph,
    groups: IndexVec<GroupId, FusedGroup>,
    group_of: IndexVec<NodeId, Option<GroupId>>,
}

impl Partitioned {
    /// The trivial partition: each node its own group, in topological order.
    pub fn one_node_per_group(graph: Graph) -> Result<Self, CycleError> {
        let order = graph.topological_order()?;
        let mut group_of: IndexVec<NodeId, Option<GroupId>> =
            graph.nodes.iter().map(|_| None).collect();
        let mut groups = IndexVec::new();
        for id in order {
            group_of[id] = Some(groups.push(fused_group(&graph, vec![id])));
        }
        Ok(Self { graph, groups, group_of })
    }

    /// `graph` split into `groups`, each a set of nodes run as one unit.
    /// Every live node is in exactly one group, and the groups are numbered
    /// so each comes after those computing its inputs, which fails if the
    /// groups depend on each other in a cycle.
    pub fn from_groups(graph: Graph, groups: &[Vec<NodeId>]) -> Result<Self, PartitionError> {
        let order = graph.topological_order()?;
        let position: HashMap<NodeId, usize> =
            order.iter().enumerate().map(|(at, &node)| (node, at)).collect();
        let mut member_of: HashMap<NodeId, usize> = HashMap::new();
        for (group, nodes) in groups.iter().enumerate() {
            for &node in nodes {
                if !position.contains_key(&node) {
                    return Err(PartitionError::NotLive(node));
                }
                if member_of.insert(node, group).is_some() {
                    return Err(PartitionError::Repeated(node));
                }
            }
        }
        if let Some(&missing) = order.iter().find(|node| !member_of.contains_key(node)) {
            return Err(PartitionError::Missing(missing));
        }
        let ranked = group_order(&graph, groups, &member_of, &position)?;
        let mut group_of: IndexVec<NodeId, Option<GroupId>> =
            graph.nodes.iter().map(|_| None).collect();
        let mut fused = IndexVec::new();
        for group in ranked {
            let mut nodes = groups[group].clone();
            nodes.sort_by_key(|node| position[node]);
            let id = fused.next_id();
            for &node in &nodes {
                group_of[node] = Some(id);
            }
            let _pushed: GroupId = fused.push(fused_group(&graph, nodes));
        }
        Ok(Self { graph, groups: fused, group_of })
    }

    /// The partitioned graph.
    pub const fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The groups, each after the groups computing its inputs.
    pub const fn groups(&self) -> &IndexVec<GroupId, FusedGroup> {
        &self.groups
    }

    /// The group holding `node`, or `None` if `node` is not a live node.
    pub fn group_of(&self, node: NodeId) -> Option<GroupId> {
        self.group_of.get(node).copied().flatten()
    }
}

/// Why groups do not partition a graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum PartitionError {
    /// The graph has a cycle.
    #[error(transparent)]
    Cycle(#[from] CycleError),
    /// A node is in more than one group.
    #[error("node {0:?} is in more than one group")]
    Repeated(NodeId),
    /// A live node is in no group.
    #[error("node {0:?} is in no group")]
    Missing(NodeId),
    /// A group lists a node that is erased or not in the graph.
    #[error("node {0:?} is not a live node of the graph")]
    NotLive(NodeId),
    /// The groups depend on each other in a cycle.
    #[error("the groups depend on each other in a cycle")]
    GroupCycle,
}

/// The groups' indices in an order where each comes after the groups
/// computing its inputs (Kahn's algorithm), the group whose first node
/// comes first in `position` going first among those ready.
fn group_order(
    graph: &Graph,
    groups: &[Vec<NodeId>],
    member_of: &HashMap<NodeId, usize>,
    position: &HashMap<NodeId, usize>,
) -> Result<Vec<usize>, PartitionError> {
    let mut readers: Vec<HashSet<usize>> = groups.iter().map(|_| HashSet::new()).collect();
    let mut pending = vec![0_usize; groups.len()];
    for (group, nodes) in groups.iter().enumerate() {
        let producers: HashSet<usize> = nodes
            .iter()
            .filter_map(|&node| graph.node(node))
            .flat_map(|node| node.operands().iter())
            .filter_map(|&value| member_of.get(&graph.producer(value)?).copied())
            .filter(|&producer| producer != group)
            .collect();
        pending[group] = producers.len();
        for producer in producers {
            let _new = readers[producer].insert(group);
        }
    }
    let first = |group: usize| groups[group].iter().map(|node| position[node]).min();
    let mut ready: BinaryHeap<Reverse<(Option<usize>, usize)>> =
        (0..groups.len()).filter(|&g| pending[g] == 0).map(|g| Reverse((first(g), g))).collect();
    let mut ranked = Vec::with_capacity(groups.len());
    while let Some(Reverse((_, group))) = ready.pop() {
        ranked.push(group);
        for &reader in &readers[group] {
            pending[reader] -= 1;
            if pending[reader] == 0 {
                ready.push(Reverse((first(reader), reader)));
            }
        }
    }
    if ranked.len() == groups.len() { Ok(ranked) } else { Err(PartitionError::GroupCycle) }
}

/// The group of `nodes`, in topological order: the values they read and
/// do not compute, each once, and those they compute that are read
/// outside them or are graph outputs.
fn fused_group(graph: &Graph, nodes: Vec<NodeId>) -> FusedGroup {
    let members: HashSet<NodeId> = nodes.iter().copied().collect();
    let computed_here =
        |value: ValueId| graph.producer(value).is_some_and(|p| members.contains(&p));
    let mut inputs: Vec<ValueId> = Vec::new();
    let mut outputs: Vec<ValueId> = Vec::new();
    for node in nodes.iter().filter_map(|&id| graph.node(id)) {
        for &operand in node.operands() {
            if !computed_here(operand) && !inputs.contains(&operand) {
                inputs.push(operand);
            }
        }
        for &value in node.outputs() {
            let read_outside = graph.uses(value).iter().any(|reader| match reader {
                Use::Operand { node, .. } => !members.contains(node),
                Use::GraphOutput(_) => true,
            });
            if read_outside {
                outputs.push(value);
            }
        }
    }
    let kind = group_kind(nodes.iter().filter_map(|&id| graph.node(id)).map(Node::op));
    FusedGroup { nodes, inputs, outputs, kind }
}

/// The kind of a group of `ops`: that of its one op that is not injective,
/// if it has one; else injective, or a view if every op is one.
fn group_kind<'o>(ops: impl Iterator<Item = &'o Op>) -> GroupKind {
    let kinds: Vec<GroupKind> = ops.map(GroupKind::of).collect();
    let anchor =
        kinds.iter().copied().find(|kind| !matches!(kind, GroupKind::Injective | GroupKind::View));
    match anchor {
        Some(kind) => kind,
        None if kinds.iter().all(|&kind| kind == GroupKind::View) => GroupKind::View,
        None => GroupKind::Injective,
    }
}
