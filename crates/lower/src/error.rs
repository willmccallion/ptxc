//! `LowerError`: a group that cannot be lowered.

use ptxc_graph::{GroupId, NodeId};
use ptxc_loop::ComputeError;
use thiserror::Error;

/// A group that cannot be lowered to compute definitions.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LowerError {
    /// A group of several nodes, which needs epilogue fusion.
    #[error("group {0:?} holds several nodes, which cannot be lowered yet")]
    Fused(GroupId),
    /// An op with no lowering yet.
    #[error("`{op}` cannot be lowered yet")]
    Unsupported {
        /// The op's mnemonic.
        op: &'static str,
    },
    /// A node whose operands its op's type rule would have rejected.
    #[error("{0:?} has operands its op does not take")]
    Operands(NodeId),
    /// A definition built for a node that `ComputeBuilder` rejected, which
    /// is a bug in lowering.
    #[error("lowering {node:?} built an ill-formed definition: {source}")]
    Compute {
        /// The node lowered.
        node: NodeId,
        /// Why the definition was rejected.
        source: ComputeError,
    },
}
