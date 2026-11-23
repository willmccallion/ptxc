//! `ScheduleError`: why a primitive cannot be applied.

use super::loop_tree::LoopId;
use super::primitive::ThreadAxis;
use crate::compute::VarName;
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
    /// A reduction loop bound to a block axis, which cannot share a sum.
    #[error("reduction loop {0} cannot be bound to a block axis")]
    BlockBoundReduce(VarName),
    /// A reduction loop vectorized.
    #[error("reduction loop {0} cannot be vectorized")]
    VectorizedReduce(VarName),
}
