//! `Replacement`: a template resolved against a match, ready to build.

use crate::pass::PassError;
use ptxc_core::HostTensor;
use ptxc_graph::{Graph, Layout, Op, RewriteError, Rewriter, TensorType, TypeRule, ValueId};

/// What replaces a match: its template resolved against the graph, each
/// node it builds already accepted by its op's type rule, so a rule that
/// does not apply leaves the graph untouched.
#[derive(Clone, Debug)]
pub(crate) enum Replacement {
    /// A value already in the graph.
    Existing(ValueId),
    /// A new node of `op` reading `operands`, its one output of type `ty`.
    Node {
        /// The op.
        op: Op,
        /// What the node reads.
        operands: Vec<Self>,
        /// The output's type, as the op's type rule gives it.
        ty: TensorType,
    },
    /// A new constant.
    Constant(HostTensor),
}

impl Replacement {
    /// A node of `op` reading `operands`, which have `types`; `None` if
    /// the op's type rule refuses them.
    pub(super) fn node(op: Op, operands: Vec<Self>, types: &[TensorType]) -> Option<Self> {
        let types: Vec<&TensorType> = types.iter().collect();
        let ty = op.infer(&types).ok()?.into_iter().next()?;
        Some(Self::Node { op, operands, ty })
    }

    /// The type of the value the replacement gives.
    pub(super) fn ty(&self, graph: &Graph) -> TensorType {
        match self {
            Self::Existing(value) => graph.value(*value).ty().clone(),
            Self::Node { ty, .. } => ty.clone(),
            Self::Constant(tensor) => {
                TensorType::new(tensor.dtype(), tensor.shape().clone(), Layout::RowMajor)
            }
        }
    }

    /// Adds the replacement's nodes and constants through `rewriter`, and
    /// returns the value it gives: `None` only for an op with no output,
    /// which no template builds.
    pub(crate) fn build(self, rewriter: &mut Rewriter<'_>) -> Result<Option<ValueId>, PassError> {
        match self {
            Self::Existing(value) => Ok(Some(value)),
            Self::Node { op, operands, .. } => {
                let mut values = Vec::with_capacity(operands.len());
                for operand in operands {
                    let Some(value) = operand.build(rewriter)? else { return Ok(None) };
                    values.push(value);
                }
                let outputs = rewriter.add_node(op, values).map_err(RewriteError::from)?;
                Ok(outputs.first().copied())
            }
            Self::Constant(tensor) => Ok(Some(rewriter.add_constant(tensor))),
        }
    }
}
