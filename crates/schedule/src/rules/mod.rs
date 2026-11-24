//! The schedule rules, one per kernel family.

mod conv_implicit_gemm;
pub(crate) mod elementwise;
mod matmul_mma;
pub(crate) mod matmul_shape;
pub(crate) mod matmul_simt;
pub(crate) mod reduction;

use ptxc_loop::{LoopId, Open, Primitive, Schedule, ScheduleError, SplitFactor, ThreadAxis};

/// The threads a default block has: a multiple of the warp size that
/// leaves room for several blocks on an SM.
const BLOCK_THREADS: u64 = 256;

/// `loops`, adjacent in the nest, fused into one; `None` when there are
/// none.
fn fused(schedule: &mut Schedule<Open>, loops: &[LoopId]) -> Result<Option<LoopId>, ScheduleError> {
    match loops {
        [] => Ok(None),
        [only] => Ok(Some(*only)),
        _ => schedule.fuse(loops.to_vec()).map(Some),
    }
}

/// `id` split into an outer loop and an inner one of at most `width`
/// iterations; `None` for an empty loop, which cannot be bound.
fn split_by(
    schedule: &mut Schedule<Open>,
    id: LoopId,
    width: u64,
) -> Result<Option<[LoopId; 2]>, ScheduleError> {
    let extent = schedule.loops().loops()[id].extent.get();
    let Some(factor) = SplitFactor::new(width.min(extent)) else { return Ok(None) };
    match schedule.split(id, vec![factor])?[..] {
        [outer, inner] => Ok(Some([outer, inner])),
        _ => Err(ScheduleError::EmptySplit),
    }
}

fn bind(schedule: &mut Schedule<Open>, id: LoopId, axis: ThreadAxis) -> Result<(), ScheduleError> {
    schedule.apply(Primitive::Bind { loop_: id, axis })
}
