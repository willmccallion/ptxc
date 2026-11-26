//! `LowerError`: a group that cannot be lowered.

use ptxc_core::DType;
use ptxc_graph::{GroupId, NodeId, ValueId};
use ptxc_loop::{ComputeError, InlineError};
use thiserror::Error;

/// A group that cannot be lowered to compute definitions.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LowerError {
    /// Inlining one of a group's stages into another built an ill-formed
    /// definition, which is a bug in lowering or inlining.
    #[error("inlining the stage computing {value:?} in group {group:?} failed: {source}")]
    Inline {
        /// The group.
        group: GroupId,
        /// The value the inlined stage computes.
        value: ValueId,
        /// Why inlining failed.
        source: InlineError,
    },
    /// An op on an element type it has no lowering for yet.
    #[error("`{op}` on {} cannot be lowered yet", dtype.name())]
    UnsupportedDType {
        /// The op's mnemonic.
        op: &'static str,
        /// The element type.
        dtype: DType,
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
