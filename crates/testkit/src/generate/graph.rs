//! `Recipe`: the choices that build a fuzz case, kept so the shrinker can
//! edit them and build again.

use super::shapes::Rng;
use crate::families::values::shape;
use ptxc_core::convert::f64_to_f32;
use ptxc_core::float::{Bf16, F16};
use ptxc_core::{DType, HostTensor, TensorData};
use ptxc_graph::{Graph, GraphBuilder, Layout, Op, TensorType, ValueId, ValueName};

/// Where a node's operand comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Operand {
    /// A graph input, by position.
    Input(usize),
    /// An earlier node's only output, by position.
    Node(usize),
}

/// One node: an op and its operands.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Node {
    /// The op.
    pub op: Op,
    /// Its operands.
    pub operands: Vec<Operand>,
}

/// A fuzz case as the choices that build it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Recipe {
    /// Every input's element type.
    pub dtype: DType,
    /// Each input's extents.
    pub inputs: Vec<Vec<u64>>,
    /// The nodes, in order; each reads inputs and earlier nodes.
    pub nodes: Vec<Node>,
    /// The seed of the inputs' values.
    pub data: u64,
}

/// A built case: the graph and its inputs.
#[derive(Clone, Debug)]
pub struct Case {
    /// The graph. Its outputs are the nodes no later node reads.
    pub graph: Graph,
    /// Its inputs.
    pub inputs: Vec<HostTensor>,
}

impl Recipe {
    /// The case, or `None` if the recipe has no nodes or some node's type
    /// rule rejects its operands.
    pub fn build(&self) -> Option<Case> {
        if self.nodes.is_empty() {
            return None;
        }
        let (mut builder, values) = self.builder()?;
        let read: Vec<bool> = (0..self.nodes.len())
            .map(|n| self.nodes.iter().any(|node| node.operands.contains(&Operand::Node(n))))
            .collect();
        let outputs: Vec<ValueId> =
            values.iter().zip(&read).filter(|&(_, &read)| !read).map(|(&value, _)| value).collect();
        for (n, &output) in outputs.iter().enumerate() {
            builder.set_name(output, ValueName::new(format!("y{n}")));
        }
        let graph = builder.finish(outputs).ok()?;
        Some(Case { graph, inputs: self.input_tensors() })
    }

    /// The type of each node's output, or `None` if a node is ill-typed.
    pub(crate) fn node_types(&self) -> Option<Vec<TensorType>> {
        let (builder, values) = self.builder()?;
        Some(values.iter().map(|&value| builder.value_type(value).clone()).collect())
    }

    /// A builder holding the inputs and nodes, and each node's output.
    fn builder(&self) -> Option<(GraphBuilder, Vec<ValueId>)> {
        let mut builder = GraphBuilder::new();
        let inputs: Vec<ValueId> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(n, dims)| {
                let value =
                    builder.add_input(TensorType::new(self.dtype, shape(dims), Layout::RowMajor));
                builder.set_name(value, ValueName::new(format!("x{n}")));
                value
            })
            .collect();
        let mut nodes: Vec<ValueId> = Vec::new();
        for node in &self.nodes {
            let operands = node
                .operands
                .iter()
                .map(|&operand| match operand {
                    Operand::Input(n) => inputs.get(n).copied(),
                    Operand::Node(n) => nodes.get(n).copied(),
                })
                .collect::<Option<Vec<_>>>()?;
            let outputs = builder.add_node(node.op.clone(), operands).ok()?;
            let [output] = outputs.as_slice() else { return None };
            nodes.push(*output);
        }
        Some((builder, nodes))
    }

    /// Each input's values: eighths in `[-2, 2]` for floats, exact in every
    /// float type; integers in `[-8, 8]`.
    fn input_tensors(&self) -> Vec<HostTensor> {
        let mut rng = Rng::new(self.data);
        self.inputs
            .iter()
            .map(|dims| {
                let count = usize::try_from(shape(dims).element_count().get()).unwrap();
                let draws: Vec<i32> =
                    (0..count).map(|_| i32::try_from(rng.below(33)).unwrap() - 16).collect();
                HostTensor::new(shape(dims), values(self.dtype, &draws)).unwrap()
            })
            .collect()
    }
}

/// `draws` in `[-16, 16]` as `dtype`'s values.
fn values(dtype: DType, draws: &[i32]) -> TensorData {
    let eighths = || draws.iter().map(|&d| f64::from(d) / 8.0);
    match dtype {
        DType::F16 => eighths().map(F16::from_f64).collect::<Vec<_>>().into(),
        DType::Bf16 => eighths().map(Bf16::from_f64).collect::<Vec<_>>().into(),
        DType::I32 => draws.iter().map(|&d| d / 2).collect::<Vec<_>>().into(),
        DType::F32 => eighths().map(f64_to_f32).collect::<Vec<_>>().into(),
        other => panic!("the generator never draws {other:?} inputs"),
    }
}
