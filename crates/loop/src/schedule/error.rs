//! `ScheduleError`: why a primitive cannot be applied.

use super::loop_tree::LoopId;
use super::primitive::ThreadAxis;
use crate::compute::VarName;
use ptxc_core::Dim;
use ptxc_target::ThreadsPerBlock;
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
}
