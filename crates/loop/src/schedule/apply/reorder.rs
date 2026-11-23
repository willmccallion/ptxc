//! Applies `reorder`.

use crate::schedule::error::ScheduleError;
use crate::schedule::loop_tree::{LoopId, LoopTree};

impl LoopTree {
    /// Puts `order`'s loops, wherever they are in the nest, into the
    /// positions they occupy, in `order`'s order; other loops stay put.
    pub(crate) fn reorder(&mut self, order: &[LoopId]) -> Result<(), ScheduleError> {
        let mut positions = Vec::with_capacity(order.len());
        for (index, &id) in order.iter().enumerate() {
            let position = self.position(id)?;
            if order[..index].contains(&id) {
                return Err(ScheduleError::RepeatedLoop(self.loops[id].name.clone()));
            }
            positions.push(position);
        }
        positions.sort_unstable();
        for (&position, &id) in positions.iter().zip(order) {
            self.nest[position] = id;
        }
        Ok(())
    }
}
