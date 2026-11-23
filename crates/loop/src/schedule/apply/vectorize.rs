//! Applies `vectorize`.

use crate::schedule::error::ScheduleError;
use crate::schedule::loop_tree::{LoopAnnotation, LoopId, LoopKind, LoopTree};

impl LoopTree {
    /// Marks spatial loop `target` to run as one vector operation; whether
    /// its extent is a vector width the target supports is checked when
    /// the schedule is validated.
    pub(crate) fn vectorize(&mut self, target: LoopId) -> Result<(), ScheduleError> {
        let vectorized = self.serial_loop(target)?;
        if vectorized.kind == LoopKind::Reduce {
            return Err(ScheduleError::VectorizedReduce(vectorized.name.clone()));
        }
        self.loops[target].annotation = LoopAnnotation::Vectorized;
        Ok(())
    }
}
