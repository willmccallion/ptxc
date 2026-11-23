//! The value environment: one host tensor per live SSA value.

use crate::error::{InterpError, Malformed};
use ptxc_core::HostTensor;
use ptxc_graph::{Graph, ValueDef, ValueId};
use std::collections::HashMap;

/// The tensors computed so far. Constants are read from the graph rather
/// than copied in.
#[derive(Debug)]
pub(crate) struct Env<'g> {
    graph: &'g Graph,
    computed: HashMap<ValueId, HostTensor>,
}

impl<'g> Env<'g> {
    pub(crate) fn new(graph: &'g Graph) -> Self {
        Self { graph, computed: HashMap::new() }
    }

    pub(crate) fn insert(&mut self, value: ValueId, tensor: HostTensor) {
        let _replaced: Option<HostTensor> = self.computed.insert(value, tensor);
    }

    /// The tensor of `value`, a graph input, constant or node output.
    pub(crate) fn get(&self, value: ValueId) -> Result<&HostTensor, InterpError> {
        match self.graph.value(value).def() {
            ValueDef::Constant(constant) => Ok(self.graph.constant(constant)),
            ValueDef::Input(_) | ValueDef::Output { .. } => {
                self.computed.get(&value).ok_or_else(|| Malformed::Undefined(value).into())
            }
        }
    }
}
