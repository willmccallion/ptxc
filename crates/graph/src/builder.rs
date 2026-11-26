//! `GraphBuilder`: adds nodes and infers their output types as it goes.

use crate::graph::{Graph, NodeError};
use crate::ops::Op;
use crate::tensor_type::TensorType;
use crate::value::{GraphOutputIndex, Use, ValueId, ValueName};
use ptxc_core::HostTensor;

/// Builds a graph in definition order, applying each op's type rule as its
/// node is added (D12), so a finished graph is well typed by construction.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    graph: Graph,
}

impl GraphBuilder {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds the next graph input.
    pub fn add_input(&mut self, ty: TensorType) -> ValueId {
        self.graph.push_input(ty)
    }

    /// Adds a constant tensor; its value has the tensor's type, row-major.
    pub fn add_constant(&mut self, tensor: HostTensor) -> ValueId {
        self.graph.push_constant(tensor)
    }

    /// Adds a node reading `operands` and returns the values it defines.
    pub fn add_node(&mut self, op: Op, operands: Vec<ValueId>) -> Result<Vec<ValueId>, NodeError> {
        self.graph.push_node(op, operands)
    }

    /// The type of `value`, a value of this graph.
    pub fn value_type(&self, value: ValueId) -> &TensorType {
        &self.graph.values[value].ty
    }

    /// Records the source model's name for `value`, a value of this graph.
    pub fn set_name(&mut self, value: ValueId, name: ValueName) {
        self.graph.values[value].name = Some(name);
    }

    /// The finished graph, returning `outputs` in order, each output named
    /// as its value is.
    pub fn finish(mut self, outputs: Vec<ValueId>) -> Result<Graph, NodeError> {
        for value in outputs {
            if !self.graph.is_live(value) {
                return Err(NodeError::NotLive(value));
            }
            let index: GraphOutputIndex = self.graph.outputs.push(value);
            let _: GraphOutputIndex =
                self.graph.output_names.push(self.graph.values[value].name.clone());
            self.graph.uses[value].push(Use::GraphOutput(index));
        }
        Ok(self.graph)
    }
}
