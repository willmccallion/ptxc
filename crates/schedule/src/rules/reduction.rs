//! Reductions: one block per output row, its threads striding along the
//! row and combining their partial results in shared memory.

use super::{BLOCK_THREADS, bind, fused, split_by};
use ptxc_loop::{LoopId, LoopKind, Open, Schedule, ScheduleError, ThreadAxis};

/// Fuses the spatial loops into rows, each bound to a block, and the
/// reduction loops into one, split so `threadIdx.x` takes consecutive
/// elements and a tree combines the threads' partial results. An empty
/// output, or an empty row, stays serial, as an empty loop cannot be bound.
pub(crate) fn one_block_per_row(schedule: &mut Schedule<Open>) -> Result<(), ScheduleError> {
    let (spatial, reduced) = by_kind(schedule);
    if let Some(rows) = fused(schedule, &spatial)? {
        // No output rows, no work: the kernel stays serial.
        if extent(schedule, rows) == 0 {
            return Ok(());
        }
        bind(schedule, rows, ThreadAxis::BlockX)?;
    }
    let Some(row) = fused(schedule, &reduced)? else { return Ok(()) };
    let Some([_, lanes]) = split_by(schedule, row, BLOCK_THREADS)? else { return Ok(()) };
    bind(schedule, lanes, ThreadAxis::ThreadX)
}

/// The nest's spatial loops and reduction loops, each in nest order.
fn by_kind(schedule: &Schedule<Open>) -> (Vec<LoopId>, Vec<LoopId>) {
    let loops = schedule.loops().loops();
    schedule.loops().nest().iter().partition(|&&id| loops[id].kind == LoopKind::Spatial)
}

fn extent(schedule: &Schedule<Open>, id: LoopId) -> u64 {
    schedule.loops().loops()[id].extent.get()
}
