//! `Interpreter`: walks a graph in topological order and implements `Evaluate`.

use crate::env::Env;
use crate::error::InterpError;
use crate::eval;
use ptxc_core::{Evaluate, HostTensor};
use ptxc_graph::text::format_type;
use ptxc_graph::{Graph, Layout, TensorType};

/// The reference interpreter for one graph.
#[derive(Debug)]
pub struct Interpreter<'g> {
    graph: &'g Graph,
}

impl<'g> Interpreter<'g> {
    /// An interpreter for `graph`.
    pub const fn new(graph: &'g Graph) -> Self {
        Self { graph }
    }

    /// Binds the inputs, checking each against its declared type.
    fn bind_inputs(&self, env: &mut Env<'g>, inputs: &[HostTensor]) -> Result<(), InterpError> {
        let declared = self.graph.inputs();
        let expected = declared.iter().len();
        if expected != inputs.len() {
            return Err(InterpError::InputCount { expected, actual: inputs.len() });
        }
        for (index, (&value, tensor)) in declared.iter().zip(inputs).enumerate() {
            let expected = self.graph.value(value).ty();
            let actual = TensorType::new(tensor.dtype(), tensor.shape().clone(), Layout::RowMajor);
            if &actual != expected {
                let (expected, actual) = (format_type(expected), format_type(&actual));
                return Err(InterpError::InputType { index, expected, actual });
            }
            env.insert(value, tensor.clone());
        }
        Ok(())
    }
}

impl Evaluate for Interpreter<'_> {
    type Error = InterpError;

    fn evaluate(&mut self, inputs: &[HostTensor]) -> Result<Vec<HostTensor>, InterpError> {
        let mut env = Env::new(self.graph);
        self.bind_inputs(&mut env, inputs)?;
        for node in self.graph.topological_order()? {
            let Some(node) = self.graph.node(node) else { continue };
            let types: Vec<&TensorType> =
                node.outputs().iter().map(|&value| self.graph.value(value).ty()).collect();
            let results = {
                let operands = node
                    .operands()
                    .iter()
                    .map(|&value| env.get(value))
                    .collect::<Result<Vec<_>, _>>()?;
                eval::evaluate(node.op(), &operands, &types)?
            };
            for (&value, tensor) in node.outputs().iter().zip(results) {
                env.insert(value, tensor);
            }
        }
        self.graph.outputs().iter().map(|&value| env.get(value).cloned()).collect()
    }
}
