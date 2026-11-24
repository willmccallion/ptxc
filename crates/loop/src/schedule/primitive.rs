//! `Primitive`: one schedule step.

use super::loop_tree::LoopId;
use crate::compute::TensorParamId;
use crate::stmt::Swizzle;
use ptxc_target::MmaInstruction;

/// A split's factor for one of the inner loops it makes; never zero.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SplitFactor(u64);

impl SplitFactor {
    /// `factor`, if it is not zero.
    pub const fn new(factor: u64) -> Option<Self> {
        if factor == 0 { None } else { Some(Self(factor)) }
    }

    /// The factor.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A software pipeline's stages: how many iterations' tiles shared memory
/// holds at once; at least 2.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StageCount(u32);

impl StageCount {
    /// `stages`, if it is at least 2.
    pub const fn new(stages: u32) -> Option<Self> {
        if stages >= 2 { Some(Self(stages)) } else { None }
    }

    /// The stages.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A CUDA grid or block axis a loop can be bound to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ThreadAxis {
    /// `blockIdx.x`.
    BlockX,
    /// `blockIdx.y`.
    BlockY,
    /// `blockIdx.z`.
    BlockZ,
    /// `threadIdx.x`.
    ThreadX,
    /// `threadIdx.y`.
    ThreadY,
    /// `threadIdx.z`.
    ThreadZ,
}

impl ThreadAxis {
    /// Whether the axis numbers blocks rather than threads in a block.
    pub const fn is_block(self) -> bool {
        matches!(self, Self::BlockX | Self::BlockY | Self::BlockZ)
    }

    /// The axis as CUDA names it, such as `threadIdx.x`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::BlockX => "blockIdx.x",
            Self::BlockY => "blockIdx.y",
            Self::BlockZ => "blockIdx.z",
            Self::ThreadX => "threadIdx.x",
            Self::ThreadY => "threadIdx.y",
            Self::ThreadZ => "threadIdx.z",
        }
    }
}

/// One step that transforms a loop tree, as TVM's schedule primitives.
/// The compute-at and layout steps come with M3's tensor-core schedules.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Primitive {
    /// Splits a loop into an outer loop and one inner loop per factor,
    /// innermost last; the outer extent rounds up.
    Split {
        /// The loop split.
        loop_: LoopId,
        /// The inner loops' extents, outermost first.
        factors: Vec<SplitFactor>,
    },
    /// Puts the listed loops, which may be anywhere in the nest, into the
    /// positions they occupy, in the listed order.
    Reorder {
        /// The loops, in their new order.
        order: Vec<LoopId>,
    },
    /// Fuses adjacent loops, outermost first, into one.
    Fuse {
        /// The loops fused, each directly inside the one before.
        loops: Vec<LoopId>,
    },
    /// Runs a loop's iterations on a grid or block axis.
    Bind {
        /// The loop bound.
        loop_: LoopId,
        /// The axis.
        axis: ThreadAxis,
    },
    /// Unrolls a loop fully.
    Unroll {
        /// The loop unrolled.
        loop_: LoopId,
    },
    /// Runs a loop's iterations as one vector operation.
    Vectorize {
        /// The loop vectorized.
        loop_: LoopId,
    },
    /// Computes the tile the innermost loops `m`, `n` and `k` iterate with
    /// one `mma` per warp, as TVM's `tensorize` replaces a loop nest with
    /// an intrinsic. The block's `threadIdx.x` becomes the warp's 32 lanes.
    Tensorize {
        /// The tile's rows.
        m: LoopId,
        /// The tile's columns.
        n: LoopId,
        /// The tile's depth, a reduction loop.
        k: LoopId,
        /// The `mma` the tile becomes.
        instruction: MmaInstruction,
    },
    /// Stages the tile of an input that each iteration of a serial loop
    /// reads in shared memory, as TVM's `cache_read` with `compute_at`.
    CacheRead {
        /// The input staged.
        tensor: TensorParamId,
        /// The loop whose every iteration fills the tile.
        at: LoopId,
    },
    /// Places the bytes of an input's staged tile by an XOR swizzle, as
    /// CUTLASS's shared-memory layouts do, so the rows an `ldmatrix` or a
    /// copy touches at once fall in different banks.
    Swizzle {
        /// The staged input.
        tensor: TensorParamId,
        /// The swizzle.
        swizzle: Swizzle,
    },
    /// Fills the tiles staged at a serial loop `stages - 1` iterations
    /// ahead, as CUTLASS's multistage mainloop does: shared memory holds
    /// `stages` iterations' tiles, and an iteration computes on its own
    /// while the next ones' copies are in flight.
    Pipeline {
        /// The loop the tiles are staged at.
        loop_: LoopId,
        /// The iterations' tiles held at once.
        stages: StageCount,
    },
}
