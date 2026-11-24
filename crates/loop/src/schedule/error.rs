//! `ScheduleError`: why a primitive cannot be applied.

use super::loop_tree::LoopId;
use super::primitive::ThreadAxis;
use crate::compute::{TensorName, TensorParamId, VarName};
use ptxc_core::{ByteSize, Dim};
use ptxc_target::{SmemBytes, ThreadsPerBlock};
use thiserror::Error;

/// A primitive the loop tree rejects. A rejected primitive changes nothing.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ScheduleError {
    /// A loop that is not in the nest: unknown, or split or fused away.
    #[error("loop {0} is not in the loop nest")]
    NotInNest(VarName),
    /// A loop id that no loop has.
    #[error("no loop has id {0:?}")]
    UnknownLoop(LoopId),
    /// A loop that is already bound, unrolled or vectorized.
    #[error("loop {0} is already annotated, so it cannot be split, fused or annotated again")]
    Annotated(VarName),
    /// A split without factors.
    #[error("a split needs at least one factor")]
    EmptySplit,
    /// Extents whose product overflows `u64`.
    #[error("the product of the extents overflows")]
    ExtentOverflow,
    /// A fuse of fewer than two loops.
    #[error("a fuse needs at least two loops")]
    FuseTooFew,
    /// Fused loops that are not each directly inside the one before.
    #[error("loop {0} is not directly inside the loop before it")]
    NotAdjacent(VarName),
    /// Fused loops of which some are spatial and some reduce.
    #[error("loop {0} is not the same kind as the loops fused with it")]
    MixedKinds(VarName),
    /// A loop listed twice.
    #[error("loop {0} is listed twice")]
    RepeatedLoop(VarName),
    /// A thread axis bound twice.
    #[error("{} is already bound", .0.name())]
    AxisTaken(ThreadAxis),
    /// An empty loop bound to an axis, which would launch no threads or
    /// blocks, a launch CUDA rejects.
    #[error("loop {0} is empty and cannot be bound to an axis")]
    EmptyBound(VarName),
    /// A reduction loop bound to a block axis, which cannot share a sum.
    #[error("reduction loop {0} cannot be bound to a block axis")]
    BlockBoundReduce(VarName),
    /// A loop bound to an axis longer than the target allows.
    #[error("{} has {} iterations, more than the target's {limit}", .axis.name(), .extent.get())]
    AxisTooLarge {
        /// The axis.
        axis: ThreadAxis,
        /// The bound loop's extent.
        extent: Dim,
        /// The target's limit for the axis.
        limit: u64,
    },
    /// More threads per block than the target allows.
    #[error("{threads} threads per block, more than the target's {}", .limit.get())]
    TooManyThreads {
        /// The block's threads.
        threads: u64,
        /// The target's limit.
        limit: ThreadsPerBlock,
    },
    /// A vectorized loop of a width vector instructions do not have.
    #[error("vectorized loop {name} has {} iterations; vectors have 2 or 4", .extent.get())]
    VectorWidth {
        /// The loop.
        name: VarName,
        /// Its extent.
        extent: Dim,
    },
    /// A vectorized loop with a serial loop inside it.
    #[error("vectorized loop {0} is not the innermost serial loop")]
    VectorNotInnermost(VarName),
    /// A vectorized loop a guard reads, whose lanes would need masking.
    #[error("vectorized loop {0} is read by a guard")]
    VectorGuarded(VarName),
    /// Reduction loops bound to more than one thread axis.
    #[error("reduction loops are bound to more than one thread axis")]
    CrossThreadAxes,
    /// A serial spatial loop inside a reduction shared between threads,
    /// which would need one partial result per spatial point per thread.
    #[error("spatial loop {0} is inside a reduction shared between threads")]
    SpatialInsideCrossThread(VarName),
    /// A reduction loop vectorized.
    #[error("reduction loop {0} cannot be vectorized")]
    VectorizedReduce(VarName),
    /// An input id the compute definition does not have.
    #[error("the compute definition has no input {0:?}")]
    UnknownTensor(TensorParamId),
    /// An input staged twice.
    #[error("input {0} is already staged in shared memory")]
    AlreadyCached(TensorName),
    /// A staged input no load reads: unread, or read only as a gather
    /// index.
    #[error("input {0} has no load to stage: it is unread or read only as a gather index")]
    CacheUnread(TensorName),
    /// A staged input read at more than one index.
    #[error("input {0} is read at more than one index, so it has no one tile to stage")]
    CacheLoads(TensorName),
    /// A staged input read through a gather index, whose tile depends on
    /// data.
    #[error("input {0} is read through a gather index, so its tile depends on data")]
    CacheGathered(TensorName),
    /// An input staged at or inside the reduction's loops but also read
    /// outside the reduction's body, where those loops do not run.
    #[error("input {0} is staged inside the reduction's loops but also read outside its body")]
    CacheOutsideReduction(TensorName),
    /// A staged input whose index is not a sum of the loops its tile spans
    /// with non-negative coefficients, so the tile is not a box from its
    /// first point.
    #[error("input {0}'s index does not grow affinely with the loops its tile spans")]
    CacheNotAffine(TensorName),
    /// A stage at a loop that is not a serial loop of the nest: bound or
    /// annotated, or split or fused away after the stage.
    #[error("input {tensor} is staged at loop {at}, which is not a serial loop of the nest")]
    CacheNotSerial {
        /// The input staged.
        tensor: TensorName,
        /// The loop it is staged at.
        at: VarName,
    },
    /// A stage in a kernel whose reduction is shared between threads, which
    /// lowering does not combine with one yet.
    #[error("input {0} cannot be staged in a reduction shared between threads")]
    CacheCrossThread(TensorName),
    /// A pipelined loop at which no input is staged, so there is nothing to
    /// fill ahead.
    #[error("loop {0} is pipelined, but no input is staged at it")]
    PipelineUnstaged(VarName),
    /// A tensorized loop whose extent is not the instruction's.
    #[error("loop {loop_} has {actual:?} iterations, but the mma's tile has {expected:?}")]
    TensorizeExtent {
        /// The loop.
        loop_: VarName,
        /// The instruction's extent.
        expected: Dim,
        /// The loop's.
        actual: Dim,
    },
    /// A tensorized row or column loop that is a reduction, or a depth loop
    /// that is not.
    #[error("loop {0} is not the kind of loop its place in the tile needs")]
    TensorizeKind(VarName),
    /// A second `tensorize`.
    #[error("the nest is already tensorized")]
    AlreadyTensorized,
    /// Tensorized loops that are not the innermost serial loops, rows, then
    /// columns, then depth.
    #[error("the tensorized loops are not the innermost serial loops, in order")]
    TensorizeNotInnermost,
    /// A tensorized tile with no reduction loop outside it, whose steps the
    /// accumulators persist across.
    #[error("no reduction loop encloses the tensorized tile")]
    TensorizeNoStep,
    /// A reduction loop inside one of the loops over a warp's tiles, where
    /// each step loads every tile's fragments before multiplying them.
    #[error("a reduction loop lies inside a loop over the warp's tiles")]
    TensorizeStepInsideTiles,
    /// A tensorized definition that is not a sum of products of two loads
    /// of the instruction's types.
    #[error("the definition is not a sum of products the mma computes")]
    TensorizeNotProduct,
    /// An operand or output whose index does not step one element per
    /// iteration of the tile's loops.
    #[error("{0} is not indexed one element per step of the tensorized loops")]
    TensorizeIndex(TensorName),
    /// A tensorized operand not staged in shared memory outside the tile.
    #[error("operand {0} is not staged in shared memory outside the tensorized tile")]
    TensorizeUnstaged(TensorName),
    /// Tiles past the shared memory a block has.
    #[error("the staged tiles take {bytes:?} of shared memory; a block has {limit:?}")]
    SharedTooLarge {
        /// The tiles' bytes.
        bytes: ByteSize,
        /// The block's shared memory.
        limit: SmemBytes,
    },
}
