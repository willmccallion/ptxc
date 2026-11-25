//! Applies `reindex`: an input's tile made a box in the iteration space.

use crate::compute::{ComputeDef, IterVarId, TensorParamId};
use crate::expr::ExprId;
use crate::schedule::error::ScheduleError;
use crate::schedule::loop_tree::{LoopTree, Reindexed};
use std::collections::HashSet;

impl LoopTree {
    /// Makes `tensor`'s tile, once staged, a box over `groups`, each a
    /// nonempty group of the definition's nonempty axes, no axis in two,
    /// whose extents' product fits an index. Each group's coordinate is its
    /// axes' indices fused row-major over the current nest, carried through
    /// later splits and fuses. Where the tile is staged, and whether its
    /// coordinates cover every axis the input's index reads and the tile
    /// spans, depend on the final nest, so validation checks them.
    pub(crate) fn reindex(
        &mut self,
        compute: &ComputeDef,
        tensor: TensorParamId,
        groups: &[Vec<IterVarId>],
    ) -> Result<(), ScheduleError> {
        let input = compute.inputs().get(tensor).ok_or(ScheduleError::UnknownTensor(tensor))?;
        if self.reindexed.iter().any(|found| found.tensor == tensor) {
            return Err(ScheduleError::AlreadyReindexed(input.name.clone()));
        }
        if !groups_distinct_axes(compute, groups) {
            return Err(ScheduleError::ReindexAxes(input.name.clone()));
        }
        let coordinates = groups
            .iter()
            .map(|group| self.fused_index(compute, group))
            .collect::<Result<Vec<_>, _>>()?;
        self.reindexed.push(Reindexed { tensor, groups: groups.to_vec(), coordinates });
        Ok(())
    }

    /// The indices of `group`'s axes fused row-major, simplified over the
    /// nest; the extents' product must fit an index.
    fn fused_index(
        &mut self,
        compute: &ComputeDef,
        group: &[IterVarId],
    ) -> Result<ExprId, ScheduleError> {
        let mut total: i64 = 1;
        let mut fused = self.exprs.constant(0);
        for &axis in group {
            let extent = i64::try_from(compute.iter_vars()[axis].extent.get())
                .map_err(|_| ScheduleError::ExtentOverflow)?;
            total = total.checked_mul(extent).ok_or(ScheduleError::ExtentOverflow)?;
            let extent = self.exprs.constant(extent);
            let scaled = self.exprs.mul(fused, extent);
            fused = self.exprs.add(scaled, self.axes[axis]);
        }
        Ok(self.simplified(fused))
    }
}

/// Whether `groups` is at least one nonempty group of `compute`'s nonempty
/// axes, no axis in two.
fn groups_distinct_axes(compute: &ComputeDef, groups: &[Vec<IterVarId>]) -> bool {
    let mut seen = HashSet::new();
    let usable =
        |axis: IterVarId| compute.iter_vars().get(axis).is_some_and(|found| found.extent.get() > 0);
    !groups.is_empty()
        && groups.iter().all(|group| {
            !group.is_empty() && group.iter().all(|&axis| usable(axis) && seen.insert(axis))
        })
}
