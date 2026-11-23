//! Applies `bind` to block and thread axes.

use crate::schedule::error::ScheduleError;
use crate::schedule::loop_tree::{LoopAnnotation, LoopId, LoopKind, LoopTree};
use crate::schedule::primitive::ThreadAxis;

impl LoopTree {
    /// Runs `target`'s iterations on `axis`. Each axis is bound once, and a
    /// reduction loop may be bound only to a thread axis, whose threads
    /// combine their partial results.
    pub(crate) fn bind(&mut self, target: LoopId, axis: ThreadAxis) -> Result<(), ScheduleError> {
        let bound = self.serial_loop(target)?;
        if bound.extent.get() == 0 {
            return Err(ScheduleError::EmptyBound(bound.name.clone()));
        }
        if axis.is_block() && bound.kind == LoopKind::Reduce {
            return Err(ScheduleError::BlockBoundReduce(bound.name.clone()));
        }
        if self.loops.iter().any(|other| other.annotation == LoopAnnotation::Bound(axis)) {
            return Err(ScheduleError::AxisTaken(axis));
        }
        self.loops[target].annotation = LoopAnnotation::Bound(axis);
        Ok(())
    }
}
