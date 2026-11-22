//! The graph verifier: SSA, types, acyclicity and use-list invariants.

#[cfg(test)]
mod tests;

use crate::graph::Graph;
use crate::node::{NodeId, NodeOutputIndex, NodeSlot};
use crate::ops::{TypeError, TypeRule};
use crate::tensor_type::{Layout, TensorType};
use crate::traverse::CycleError;
use crate::value::{Use, ValueDef, ValueId};
use ptxc_core::IndexVec;
use thiserror::Error;

/// A broken graph invariant.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum VerifyError {
    /// A value and the input, constant or node it names as its definition
    /// disagree, so it does not have exactly one definition.
    #[error("value {0:?} and its definition disagree")]
    Definition(ValueId),
    /// A constant's value has a different type from the tensor.
    #[error("constant value {0:?} does not have its tensor's type")]
    ConstantType(ValueId),
    /// A node reads a value that is not live.
    #[error("node {node:?} reads {value:?}, which is not live")]
    DeadOperand {
        /// The reading node.
        node: NodeId,
        /// The value.
        value: ValueId,
    },
    /// A graph output is not live.
    #[error("graph output {0:?} is not live")]
    DeadGraphOutput(ValueId),
    /// A use list differs from the readers the nodes and outputs record.
    #[error("value {value:?} lists uses {listed:?}, but is read by {actual:?}")]
    UseList {
        /// The value.
        value: ValueId,
        /// Its use list, sorted.
        listed: Vec<Use>,
        /// Its readers, sorted.
        actual: Vec<Use>,
    },
    /// The graph is not acyclic.
    #[error(transparent)]
    Cycle(#[from] CycleError),
    /// A node's op rejects its operands.
    #[error("node {node:?}: {source}")]
    TypeRule {
        /// The node.
        node: NodeId,
        /// Why its type rule rejects the operands.
        source: TypeError,
    },
    /// A node's output types differ from what its type rule infers.
    #[error("node {node:?} has output types {recorded:?}, but its type rule gives {inferred:?}")]
    OutputTypes {
        /// The node.
        node: NodeId,
        /// The types its outputs carry.
        recorded: Vec<TensorType>,
        /// The types its type rule infers.
        inferred: Vec<TensorType>,
    },
}

impl Graph {
    /// Checks every graph invariant: each value has exactly one definition,
    /// every read is of a live value, the use lists match the readers, the
    /// graph is acyclic, and every node's output types are what its type rule
    /// infers.
    pub fn verify(&self) -> Result<(), VerifyError> {
        self.verify_definitions()?;
        self.verify_reads()?;
        self.verify_use_lists()?;
        let _: Vec<NodeId> = self.topological_order()?;
        self.verify_types()
    }

    fn verify_definitions(&self) -> Result<(), VerifyError> {
        for (id, value) in self.values.iter_enumerated() {
            let agrees = match value.def {
                ValueDef::Input(index) => self.inputs.get(index) == Some(&id),
                ValueDef::Constant(constant) => self.constants.get(constant).is_some(),
                ValueDef::Output { node, index } => self.output_agrees(id, node, index),
            };
            if !agrees {
                return Err(VerifyError::Definition(id));
            }
            if let ValueDef::Constant(constant) = value.def {
                let tensor = &self.constants[constant];
                let ty = TensorType::new(tensor.dtype(), tensor.shape().clone(), Layout::RowMajor);
                if value.ty != ty {
                    return Err(VerifyError::ConstantType(id));
                }
            }
        }
        self.verify_definitions_point_back()
    }

    /// A value defined by an erased node agrees with it: it is dead, and the
    /// use-list check confirms nothing reads it.
    fn output_agrees(&self, value: ValueId, node: NodeId, index: NodeOutputIndex) -> bool {
        match self.nodes.get(node) {
            Some(NodeSlot::Live(node)) => node.outputs.get(index) == Some(&value),
            Some(NodeSlot::Erased) => true,
            None => false,
        }
    }

    fn verify_definitions_point_back(&self) -> Result<(), VerifyError> {
        for (index, &value) in self.inputs.iter_enumerated() {
            if self.values.get(value).map(|value| value.def) != Some(ValueDef::Input(index)) {
                return Err(VerifyError::Definition(value));
            }
        }
        for (node, contents) in self.nodes() {
            for (index, &value) in contents.outputs.iter_enumerated() {
                let def = self.values.get(value).map(|value| value.def);
                if def != Some(ValueDef::Output { node, index }) {
                    return Err(VerifyError::Definition(value));
                }
            }
        }
        Ok(())
    }

    fn verify_reads(&self) -> Result<(), VerifyError> {
        for (node, contents) in self.nodes() {
            if let Some(&value) = contents.operands.iter().find(|&&value| !self.is_live(value)) {
                return Err(VerifyError::DeadOperand { node, value });
            }
        }
        match self.outputs.iter().find(|&&value| !self.is_live(value)) {
            Some(&value) => Err(VerifyError::DeadGraphOutput(value)),
            None => Ok(()),
        }
    }

    fn verify_use_lists(&self) -> Result<(), VerifyError> {
        let mut readers: IndexVec<ValueId, Vec<Use>> =
            self.values.iter().map(|_| Vec::new()).collect();
        for (node, contents) in self.nodes() {
            for (index, &value) in contents.operands.iter_enumerated() {
                readers[value].push(Use::Operand { node, index });
            }
        }
        for (index, &value) in self.outputs.iter_enumerated() {
            readers[value].push(Use::GraphOutput(index));
        }
        for (value, readers) in readers.iter_enumerated() {
            let mut listed = self.uses[value].clone();
            let mut actual = readers.clone();
            listed.sort_unstable();
            actual.sort_unstable();
            if listed != actual {
                return Err(VerifyError::UseList { value, listed, actual });
            }
        }
        Ok(())
    }

    fn verify_types(&self) -> Result<(), VerifyError> {
        for (node, contents) in self.nodes() {
            let operand_types: Vec<&TensorType> =
                contents.operands.iter().map(|&value| &self.values[value].ty).collect();
            let inferred = contents
                .op
                .infer(&operand_types)
                .map_err(|source| VerifyError::TypeRule { node, source })?;
            let recorded: Vec<TensorType> =
                contents.outputs.iter().map(|&value| self.values[value].ty.clone()).collect();
            if recorded != inferred {
                return Err(VerifyError::OutputTypes { node, recorded, inferred });
            }
        }
        Ok(())
    }
}
