//! The mma instructions (shape and operand types) a target supports.

use ptxc_core::{DType, Dim};

/// The `m`, `n` and `k` extents of one `mma` instruction: `A` is `m x k`,
/// `B` is `k x n`, and the accumulator is `m x n`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MmaShape {
    /// Rows of `A` and the accumulator.
    pub m: Dim,
    /// Columns of `B` and the accumulator.
    pub n: Dim,
    /// The contracted extent.
    pub k: Dim,
}

/// One warp-level `mma.sync` variant: its shape, the element type of both
/// multiplicands, and the accumulator's.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MmaInstruction {
    /// The shape.
    pub shape: MmaShape,
    /// The element type of `A` and `B`.
    pub multiplicands: DType,
    /// The element type of the accumulator and result.
    pub accumulator: DType,
}

/// `sm_89`'s mma instructions. Rows are added with tensor-core schedules (M3).
pub(crate) const SM_89: &[MmaInstruction] = &[];
