//! `Schedule<S>`: a loop tree being transformed (`Open`) or checked (`Validated`).

mod apply;
mod error;
mod loop_tree;
mod primitive;
mod trace;
mod validate;
mod validate_caches;

pub use error::ScheduleError;
pub use loop_tree::{CacheRead, Guard, Loop, LoopAnnotation, LoopId, LoopKind, LoopTree};
pub use primitive::{Primitive, SplitFactor, ThreadAxis};
pub use trace::Trace;
pub use validate::Validated;

use crate::compute::ComputeDef;
use std::sync::Arc;

/// A schedule being built: primitives can still be applied.
#[derive(Clone, Copy, Debug)]
pub struct Open;

/// How a compute definition's loops run, as a loop tree and the trace of
/// primitives that shaped it (D4). The state `S` says what may be done
/// next, and holds what reaching it established.
#[derive(Clone, Debug)]
pub struct Schedule<S> {
    compute: Arc<ComputeDef>,
    loops: LoopTree,
    trace: Trace,
    state: S,
}

impl Schedule<Open> {
    /// The default nest: one serial loop per axis, spatial axes outermost.
    pub fn new(compute: Arc<ComputeDef>) -> Self {
        let loops = LoopTree::new(&compute);
        Self { compute, loops, trace: Trace::new(), state: Open }
    }

    /// The schedule `trace` builds from the default nest.
    pub fn replay(compute: Arc<ComputeDef>, trace: &Trace) -> Result<Self, ScheduleError> {
        let mut schedule = Self::new(compute);
        for step in trace.steps() {
            schedule.apply(step.clone())?;
        }
        Ok(schedule)
    }

    /// Applies `step` and records it; a rejected step changes nothing.
    pub fn apply(&mut self, step: Primitive) -> Result<(), ScheduleError> {
        match step {
            Primitive::Split { loop_, factors } => self.split(loop_, factors).map(drop),
            Primitive::Fuse { loops } => self.fuse(loops).map(drop),
            Primitive::Reorder { ref order } => {
                self.record(step.clone(), |tree| tree.reorder(order))
            }
            Primitive::Bind { loop_, axis } => self.record(step, |tree| tree.bind(loop_, axis)),
            Primitive::Unroll { loop_ } => self.record(step, |tree| tree.unroll(loop_)),
            Primitive::Vectorize { loop_ } => self.record(step, |tree| tree.vectorize(loop_)),
            Primitive::CacheRead { tensor, at } => {
                let compute = Arc::clone(&self.compute);
                self.record(step, |tree| tree.cache_read(&compute, tensor, at))
            }
        }
    }

    /// Splits `loop_` by `factors` (see `Primitive::Split`) and returns the
    /// parts, outermost first.
    pub fn split(
        &mut self,
        loop_: LoopId,
        factors: Vec<SplitFactor>,
    ) -> Result<Vec<LoopId>, ScheduleError> {
        let parts = self.loops.split(loop_, &factors)?;
        self.trace.push(Primitive::Split { loop_, factors });
        Ok(parts)
    }

    /// Fuses `loops` (see `Primitive::Fuse`) and returns the fused loop.
    pub fn fuse(&mut self, loops: Vec<LoopId>) -> Result<LoopId, ScheduleError> {
        let fused = self.loops.fuse(&loops)?;
        self.trace.push(Primitive::Fuse { loops });
        Ok(fused)
    }

    fn record(
        &mut self,
        step: Primitive,
        apply: impl FnOnce(&mut LoopTree) -> Result<(), ScheduleError>,
    ) -> Result<(), ScheduleError> {
        apply(&mut self.loops)?;
        self.trace.push(step);
        Ok(())
    }
}

impl<S> Schedule<S> {
    /// The compute definition scheduled.
    pub fn compute(&self) -> &ComputeDef {
        &self.compute
    }

    /// The loop nest.
    pub const fn loops(&self) -> &LoopTree {
        &self.loops
    }

    /// The primitives applied, in order.
    pub const fn trace(&self) -> &Trace {
        &self.trace
    }
}
