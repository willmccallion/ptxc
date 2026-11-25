//! Elementwise and injective kernels: every spatial loop fused into one,
//! split into blocks of threads.

use super::{fused, spread_over_blocks};
use ptxc_loop::{Open, Schedule, ScheduleError};

/// Fuses the nest into one loop spread over blocks of threads, so
/// consecutive threads write consecutive elements.
pub(crate) fn flatten_over_blocks(schedule: &mut Schedule<Open>) -> Result<(), ScheduleError> {
    let nest = schedule.loops().nest().to_vec();
    let Some(flat) = fused(schedule, &nest)? else { return Ok(()) };
    spread_over_blocks(schedule, flat)
}
