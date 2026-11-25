//! The schedule rules, one per kernel family.

pub(crate) mod conv_implicit_gemm;
pub(crate) mod elementwise;
pub(crate) mod matmul_loops;
pub(crate) mod matmul_mma;
pub(crate) mod matmul_shape;
pub(crate) mod matmul_simt;
pub(crate) mod reduction;

use ptxc_loop::{LoopId, Open, Primitive, Schedule, ScheduleError, SplitFactor, ThreadAxis};

/// The threads a default block has: a multiple of the warp size that
/// leaves room for several blocks on an SM.
const BLOCK_THREADS: u64 = 256;

/// The blocks a grid of large tiles needs to keep a GPU busy: about one per
/// SM of a mid-range Ada part, hand-picked from the sweep until tuning
/// measures each device.
pub(crate) const FULL_GRID: u64 = 32;

/// The blocks of small tiles the GPU holds at once: three on each SM, as
/// their shared memory allows, hand-picked from the sweep until tuning
/// measures each device.
pub(crate) const RESIDENT_GRID: u64 = 3 * FULL_GRID;

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

/// `id` split onto `blockIdx.x` and `threadIdx.x`, `BLOCK_THREADS` threads
/// to a block. An empty loop stays serial, as it cannot be bound.
fn spread_over_blocks(schedule: &mut Schedule<Open>, id: LoopId) -> Result<(), ScheduleError> {
    let Some([blocks, threads]) = split_by(schedule, id, BLOCK_THREADS)? else { return Ok(()) };
    bind(schedule, blocks, ThreadAxis::BlockX)?;
    bind(schedule, threads, ThreadAxis::ThreadX)
}

fn bind(schedule: &mut Schedule<Open>, id: LoopId, axis: ThreadAxis) -> Result<(), ScheduleError> {
    schedule.apply(Primitive::Bind { loop_: id, axis })
}
