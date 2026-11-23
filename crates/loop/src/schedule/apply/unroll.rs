//! Applies `unroll`.

use crate::schedule::error::ScheduleError;
use crate::schedule::loop_tree::{LoopAnnotation, LoopId, LoopTree};

impl LoopTree {
    /// Marks `target` for full unrolling.
    pub(crate) fn unroll(&mut self, target: LoopId) -> Result<(), ScheduleError> {
        let _serial = self.serial_loop(target)?;
        self.loops[target].annotation = LoopAnnotation::Unrolled;
        Ok(())
    }
}
