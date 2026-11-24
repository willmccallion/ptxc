//! Applies `pipeline`.

use crate::schedule::error::ScheduleError;
use crate::schedule::loop_tree::{LoopAnnotation, LoopId, LoopTree};
use crate::schedule::primitive::StageCount;

impl LoopTree {
    /// Marks `target` to fill the tiles staged at it `stages - 1`
    /// iterations ahead.
    pub(crate) fn pipeline(
        &mut self,
        target: LoopId,
        stages: StageCount,
    ) -> Result<(), ScheduleError> {
        let _serial = self.serial_loop(target)?;
        self.loops[target].annotation = LoopAnnotation::Pipelined(stages);
        Ok(())
    }
}
