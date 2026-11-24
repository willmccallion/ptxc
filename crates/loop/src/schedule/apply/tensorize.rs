//! Applies `tensorize`: an innermost tile computed by one `mma` per warp.

use crate::schedule::error::ScheduleError;
use crate::schedule::loop_tree::{LoopAnnotation, LoopId, LoopKind, LoopTree, Tensorized};
use crate::schedule::primitive::ThreadAxis;
use ptxc_target::MmaInstruction;

impl LoopTree {
    /// Tensorizes the tile `loops` (rows, columns, depth) with
    /// `instruction`: serial loops of its extents, rows and columns spatial
    /// and depth a reduction. `threadIdx.x` must be free for the lanes.
    /// Where the loops sit and what they index is checked by validation.
    pub(in crate::schedule) fn tensorize(
        &mut self,
        loops: [LoopId; 3],
        instruction: MmaInstruction,
    ) -> Result<(), ScheduleError> {
        if self.tensorized.is_some() {
            return Err(ScheduleError::AlreadyTensorized);
        }
        let shape = instruction.shape;
        let kinds = [LoopKind::Spatial, LoopKind::Spatial, LoopKind::Reduce];
        for ((id, expected), kind) in loops.into_iter().zip([shape.m, shape.n, shape.k]).zip(kinds)
        {
            let found = self.serial_loop(id)?;
            if found.extent != expected {
                let (loop_, actual) = (found.name.clone(), found.extent);
                return Err(ScheduleError::TensorizeExtent { loop_, expected, actual });
            }
            if found.kind != kind {
                return Err(ScheduleError::TensorizeKind(found.name.clone()));
            }
        }
        let lanes = LoopAnnotation::Bound(ThreadAxis::ThreadX);
        if self.loops.iter().any(|other| other.annotation == lanes) {
            return Err(ScheduleError::AxisTaken(ThreadAxis::ThreadX));
        }
        self.tensorized = Some(Tensorized { loops, instruction });
        Ok(())
    }
}
