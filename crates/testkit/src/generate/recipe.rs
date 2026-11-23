//! `Recipe`: the inputs and nodes that build a fuzz case, kept so the
//! shrinker can edit them and build again.

use super::connect::is_continuous;
use super::inputs::{Input, Source};
use ptxc_core::HostTensor;
use ptxc_graph::{Graph, GraphBuilder, Layout, Op, TensorType, ValueId, ValueName};

/// Where a node's operand comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Operand {
    /// A graph input, by position.
    Input(usize),
    /// An output of an earlier node, by the node's position and the
    /// output's.
    Output {
        /// The node.
        node: usize,
        /// Which of its outputs.
        index: usize,
    },
}

/// One node: an op and its operands.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Node {
    /// The op.
    pub op: Op,
    /// Its operands.
    pub operands: Vec<Operand>,
}

/// A fuzz case as the inputs and nodes that build it.
#[derive(Clone, Default, Debug)]
pub struct Recipe {
    /// The graph's inputs.
    pub inputs: Vec<Input>,
    /// The nodes, in order; each reads inputs and earlier nodes' outputs.
    pub nodes: Vec<Node>,
}

/// A built case: the graph and its inputs.
#[derive(Clone, Debug)]
pub struct Case {
    /// The graph. Its outputs are the node outputs no later node reads.
    pub graph: Graph,
    /// Its inputs.
    pub inputs: Vec<HostTensor>,
}

/// Each node's output values, in node order.
type NodeOutputs = Vec<Vec<ValueId>>;

impl Recipe {
    /// The case, or `None` if the recipe has no nodes or some node's type
    /// rule rejects its operands.
    pub fn build(&self) -> Option<Case> {
        if self.nodes.is_empty() {
            return None;
        }
        let (mut builder, produced) = self.builder()?;
        let mut outputs = Vec::new();
        for (node, values) in produced.iter().enumerate() {
            for (index, &value) in values.iter().enumerate() {
                if !self.reads(Operand::Output { node, index }) {
                    builder.set_name(value, ValueName::new(format!("y{}", outputs.len())));
                    outputs.push(value);
                }
            }
        }
        let graph = builder.finish(outputs).ok()?;
        let inputs = self.inputs.iter().map(|input| input.tensor.clone()).collect();
        Some(Case { graph, inputs })
    }

    /// Whether the case's results on the reference interpreter are
    /// comparable within tolerance: every input moderate and every op
    /// continuous, so last-bit differences stay last-bit differences.
    pub fn compares_with_reference(&self) -> bool {
        self.inputs.iter().all(|input| input.source == Source::Moderate)
            && self.nodes.iter().all(|node| is_continuous(&node.op))
    }

    /// Whether some node reads `operand`.
    pub(crate) fn reads(&self, operand: Operand) -> bool {
        self.nodes.iter().any(|node| node.operands.contains(&operand))
    }

    /// Every value a new node may read, with its type, or `None` if some
    /// node is ill-typed.
    pub(crate) fn values(&self) -> Option<Vec<(Operand, TensorType)>> {
        let (builder, produced) = self.builder()?;
        let mut values: Vec<(Operand, TensorType)> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(n, input)| (Operand::Input(n), tensor_type(&input.tensor)))
            .collect();
        for (node, outputs) in produced.iter().enumerate() {
            for (index, &value) in outputs.iter().enumerate() {
                values.push((Operand::Output { node, index }, builder.value_type(value).clone()));
            }
        }
        Some(values)
    }

    /// A builder holding the inputs and nodes, and each node's outputs.
    fn builder(&self) -> Option<(GraphBuilder, NodeOutputs)> {
        let mut builder = GraphBuilder::new();
        let mut inputs = Vec::new();
        for (n, input) in self.inputs.iter().enumerate() {
            let value = builder.add_input(tensor_type(&input.tensor));
            builder.set_name(value, ValueName::new(format!("x{n}")));
            inputs.push(value);
        }
        let mut produced: NodeOutputs = Vec::new();
        for node in &self.nodes {
            let operands = node
                .operands
                .iter()
                .map(|&operand| match operand {
                    Operand::Input(n) => inputs.get(n).copied(),
                    Operand::Output { node, index } => {
                        produced.get(node).and_then(|outputs| outputs.get(index)).copied()
                    }
                })
                .collect::<Option<Vec<_>>>()?;
            produced.push(builder.add_node(node.op.clone(), operands).ok()?);
        }
        Some((builder, produced))
    }
}

/// `tensor`'s type, row-major.
pub(crate) fn tensor_type(tensor: &HostTensor) -> TensorType {
    TensorType::new(tensor.dtype(), tensor.shape().clone(), Layout::RowMajor)
}
