//! Detects shared-memory races between barriers, and global elements two
//! threads write.

use super::RunError;
use crate::compute::TensorName;
use crate::stmt::BufferId;
use std::collections::HashMap;

/// Who touched one element since the last barrier.
#[derive(Default)]
struct Touches {
    writer: Option<usize>,
    readers: Vec<usize>,
}

/// The shared-memory accesses of the current phase, by thread. Two threads
/// touching one element in a phase, one of them writing, race: no
/// interleaving of the threads is ruled out, so the result is not defined.
#[derive(Default)]
pub(super) struct PhaseLog {
    touches: HashMap<(BufferId, i64), Touches>,
}

impl PhaseLog {
    pub(super) fn read(
        &mut self,
        buffer: BufferId,
        name: &TensorName,
        index: i64,
        thread: usize,
    ) -> Result<(), RunError> {
        let touches = self.touches.entry((buffer, index)).or_default();
        if touches.writer.is_some_and(|writer| writer != thread) {
            return Err(RunError::Race { buffer: name.clone(), index });
        }
        touches.readers.push(thread);
        Ok(())
    }

    pub(super) fn write(
        &mut self,
        buffer: BufferId,
        name: &TensorName,
        index: i64,
        thread: usize,
    ) -> Result<(), RunError> {
        let touches = self.touches.entry((buffer, index)).or_default();
        let other_writer = touches.writer.is_some_and(|writer| writer != thread);
        let other_reader = touches.readers.iter().any(|&reader| reader != thread);
        if other_writer || other_reader {
            return Err(RunError::Race { buffer: name.clone(), index });
        }
        touches.writer = Some(thread);
        Ok(())
    }

    /// Ends the phase at a barrier.
    pub(super) fn clear(&mut self) {
        self.touches.clear();
    }
}

/// Which thread of the whole grid wrote each global element.
#[derive(Default)]
pub(super) struct GlobalWriters {
    writers: HashMap<(BufferId, i64), usize>,
}

impl GlobalWriters {
    pub(super) fn write(
        &mut self,
        buffer: BufferId,
        name: &TensorName,
        index: i64,
        thread: usize,
    ) -> Result<(), RunError> {
        let writer = *self.writers.entry((buffer, index)).or_insert(thread);
        if writer == thread { Ok(()) } else { Err(RunError::Race { buffer: name.clone(), index }) }
    }
}
