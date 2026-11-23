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

/// `sm_89`'s `mma.sync` variants with a wide accumulator, as ptxc
/// accumulates (D43): f16 and bf16 at m16n8k16, and FP8 and 8-bit integers
/// at m16n8k32, from the PTX ISA's table of warp-level matrix shapes.
pub(crate) const SM_89: &[MmaInstruction] = &[
    instruction([16, 8, 16], DType::F16, DType::F32),
    instruction([16, 8, 16], DType::Bf16, DType::F32),
    instruction([16, 8, 32], DType::F8E4M3Fn, DType::F32),
    instruction([16, 8, 32], DType::F8E5M2, DType::F32),
    instruction([16, 8, 32], DType::I8, DType::I32),
    instruction([16, 8, 32], DType::U8, DType::I32),
];

const fn instruction(
    [m, n, k]: [u64; 3],
    multiplicands: DType,
    accumulator: DType,
) -> MmaInstruction {
    MmaInstruction {
        shape: MmaShape { m: Dim::new(m), n: Dim::new(n), k: Dim::new(k) },
        multiplicands,
        accumulator,
    }
}
