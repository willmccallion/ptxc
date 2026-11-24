//! Default schedules with their staged loop pipelined over a stage count a
//! test chooses rather than the one the default picks. The codegen and GPU
//! tests share them.
//!
//! NOTE: test code shared as a library. A builder handed a case it cannot
//! build has a bug, so it panics rather than returning an error (D52).

#![expect(
    clippy::unwrap_used,
    clippy::missing_panics_doc,
    reason = "test builders panic on cases they build wrong (D52)"
)]

use ptxc_loop::{ComputeDef, Primitive, Schedule, StageCount, Validated};
use ptxc_target::Target;
use std::sync::Arc;

/// `compute`'s default schedule for `target`, with the loop its first
/// staged input is staged at pipelined over `stages`.
pub fn pipelined(compute: &Arc<ComputeDef>, stages: u32, target: &Target) -> Schedule<Validated> {
    let default = ptxc_schedule::default_schedule(Arc::clone(compute), target).unwrap();
    let mut schedule = Schedule::new(Arc::clone(compute));
    for step in default.trace().steps() {
        if !matches!(step, Primitive::Pipeline { .. }) {
            schedule.apply(step.clone()).unwrap();
        }
    }
    let at = schedule.loops().caches()[0].at;
    let stages = StageCount::new(stages).unwrap();
    schedule.apply(Primitive::Pipeline { loop_: at, stages }).unwrap();
    schedule.validate(target).unwrap()
}
