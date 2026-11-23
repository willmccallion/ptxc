//! Heuristic choices for workloads that have no tuning record: an injective
//! stage spreads over blocks of threads, and a reduction gives each output
//! row a block whose threads combine through shared memory.

use crate::rules::{elementwise, reduction};
use ptxc_loop::{ComputeDef, IterKind, Schedule, ScheduleError, Validated};
use ptxc_target::Target;
use std::sync::Arc;

/// The schedule `compute` gets with no tuning record, validated for
/// `target`.
pub fn default_schedule(
    compute: Arc<ComputeDef>,
    target: &Target,
) -> Result<Schedule<Validated>, ScheduleError> {
    let reduces = compute.iter_vars().iter().any(|axis| axis.kind == IterKind::Reduce);
    let mut schedule = Schedule::new(compute);
    if reduces {
        reduction::one_block_per_row(&mut schedule)?;
    } else {
        elementwise::flatten_over_blocks(&mut schedule)?;
    }
    schedule.validate(target)
}
