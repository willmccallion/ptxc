//! Elementwise and injective kernels: every spatial loop fused into one,
//! split into blocks of threads.

use super::{BLOCK_THREADS, bind, fused, split_by};
use ptxc_loop::{Open, Schedule, ScheduleError, ThreadAxis};

/// Fuses the nest into one loop and binds it to `blockIdx.x` and
/// `threadIdx.x`, `BLOCK_THREADS` threads to a block, so consecutive
/// threads write consecutive elements. An empty output stays serial, as an
/// empty loop cannot be bound.
pub(crate) fn flatten_over_blocks(schedule: &mut Schedule<Open>) -> Result<(), ScheduleError> {
    let nest = schedule.loops().nest().to_vec();
    let Some(flat) = fused(schedule, &nest)? else { return Ok(()) };
    let Some([blocks, threads]) = split_by(schedule, flat, BLOCK_THREADS)? else { return Ok(()) };
    bind(schedule, blocks, ThreadAxis::BlockX)?;
    bind(schedule, threads, ThreadAxis::ThreadX)
}
