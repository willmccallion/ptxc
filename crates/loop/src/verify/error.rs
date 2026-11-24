//! `VerifyError`: the rules a lowered kernel can break.

use crate::compute::{TensorName, VarName};
use crate::stmt::FragmentShape;
use ptxc_core::DType;
use thiserror::Error;

/// A lowered kernel that breaks a rule codegen relies on.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum VerifyError {
    /// A variable read where nothing binds it.
    #[error("{0} is read where it is not bound")]
    UnboundVar(VarName),
    /// A store to a buffer the kernel only reads.
    #[error("{0} is an input, but is stored to")]
    StoreToInput(TensorName),
    /// A load from a buffer the kernel only writes.
    #[error("{0} is an output, but is loaded from")]
    LoadFromOutput(TensorName),
    /// A shared or local buffer used outside its `Alloc`.
    #[error("{0} is used outside its allocation")]
    NotAllocated(TensorName),
    /// An `Alloc` of a global buffer.
    #[error("{0} is global, so it is a parameter, not allocated")]
    GlobalAllocated(TensorName),
    /// A shared or local access not provably within the buffer.
    #[error("an access to {0} may leave the buffer")]
    OutOfBounds(TensorName),
    /// A barrier some threads of a block may not reach.
    #[error("a barrier is under a condition that depends on the thread")]
    DivergentBarrier,
    /// A value of the wrong type for where it goes.
    #[error("{place} takes {expected:?}, but is given {actual:?}")]
    WrongType {
        /// The store, index load or condition.
        place: String,
        /// The type it takes.
        expected: DType,
        /// The type given.
        actual: DType,
    },
    /// An operation on types it does not take.
    #[error(transparent)]
    Operation(#[from] crate::compute::ComputeError),
    /// A buffer a fragment statement takes as a fragment that is not one.
    #[error("{0} is not a fragment")]
    NotFragment(TensorName),
    /// A fragment read or written element by element, or copied to another
    /// fragment, where only the fragment statements may touch it.
    #[error("{0} is a fragment, which only the fragment statements touch")]
    FragmentAccess(TensorName),
    /// A fragment whose tile is not the shape its operation takes.
    #[error("{fragment} holds a {actual:?} tile, but its operation takes {expected:?}")]
    FragmentShape {
        /// The fragment.
        fragment: TensorName,
        /// The shape the operation takes.
        expected: FragmentShape,
        /// The fragment's shape.
        actual: FragmentShape,
    },
    /// A fragment buffer with too few elements for its tile.
    #[error("{0} has fewer elements than its tile")]
    FragmentSize(TensorName),
    /// A fragment loaded from a buffer outside shared memory.
    #[error("a fragment is loaded from {0}, which is not in shared memory")]
    TileNotShared(TensorName),
    /// A fragment statement some lanes of a warp may not reach.
    #[error("a fragment statement is under a condition that depends on the thread")]
    DivergentWarp,
    /// A fragment statement in a block whose threads do not fill whole warps.
    #[error("a fragment statement runs in a block that is not a whole number of warps")]
    PartialWarp,
}
