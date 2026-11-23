//! `FusedGroup` and `Partitioned`, a graph whose every node belongs to exactly one group.

use crate::graph::Graph;
use crate::node::{Node, NodeId};
use crate::ops::Op;
use crate::traverse::CycleError;
use crate::value::ValueId;
use ptxc_core::{IndexVec, define_index};

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
            Op::MatMul | Op::Gemm(_) | Op::QMatMul(_) => Self::MatMulEpilogue,
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
            let Some(node) = graph.node(id) else { continue };
            group_of[id] = Some(groups.push(singleton(&graph, id, node)));
        }
        Ok(Self { graph, groups, group_of })
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

/// The group holding only `node`.
fn singleton(graph: &Graph, id: NodeId, node: &Node) -> FusedGroup {
    let mut inputs: Vec<ValueId> = Vec::new();
    for &operand in node.operands() {
        if !inputs.contains(&operand) {
            inputs.push(operand);
        }
    }
    // A graph is acyclic, so every read of a node's output is outside it.
    let outputs =
        node.outputs().iter().copied().filter(|&value| !graph.uses(value).is_empty()).collect();
    FusedGroup { nodes: vec![id], inputs, outputs, kind: GroupKind::of(node.op()) }
}
