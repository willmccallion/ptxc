//! A reindexed tile, a box in the iteration space as MetaSchedule's
//! `ReIndex` stages a read: that its coordinates cover every axis varying
//! across it, the input element each of its elements holds, and whether a
//! run of its elements is adjacent in the input.

use super::cache::{Tile, tile_size};
use super::context::Lowerer;
use crate::compute::{IndexVar, IterVarId, TensorParamId};
use crate::expr::{ExprId, VarId};
use crate::schedule::ScheduleError;
use ptxc_core::Dim;
use std::collections::HashMap;

/// The space a tile is a box in.
#[derive(Clone, Debug)]
pub(super) enum TileSpace {
    /// The input's: a coordinate per input axis, its index.
    Input,
    /// The iteration space: a coordinate per group of axes, fused
    /// row-major, placing the input element the compute-level `index`
    /// reads.
    Iteration { groups: Vec<Vec<IterVarId>>, index: Vec<ExprId> },
}

impl Lowerer<'_> {
    /// Every axis a reindexed tile's index reads that varies across it,
    /// reading one of the `spanned` loops, is in one of its groups, so its
    /// coordinates place each element a load of `tensor` reads.
    pub(super) fn check_covered(
        &self,
        tensor: TensorParamId,
        space: &TileSpace,
        spanned: &[(VarId, Dim)],
    ) -> Result<(), ScheduleError> {
        let TileSpace::Iteration { groups, index } = space else { return Ok(()) };
        let compute = self.compute;
        let grouped = |axis: IterVarId| groups.iter().flatten().any(|&grouped| grouped == axis);
        let varies = |axis: IterVarId| {
            let read = self.exprs.vars(self.tree.axis_index(axis));
            read.into_iter().any(|var| spanned.iter().any(|&(spanned, _)| spanned == var))
        };
        for var in index.iter().flat_map(|&part| compute.exprs().vars(part)) {
            if let IndexVar::Iter(axis) = compute.vars()[var]
                && !grouped(axis)
                && varies(axis)
            {
                let tensor = compute.inputs()[tensor].name.clone();
                let axis = compute.iter_vars()[axis].name.clone();
                return Err(ScheduleError::ReindexUncovered { tensor, axis });
            }
        }
        Ok(())
    }

    /// The input's index of the element of `tile` at `coordinates`: the
    /// coordinates themselves, or, reindexed, the input's index with each
    /// group's axes at their places in its coordinate.
    pub(super) fn tile_read(&mut self, tile: &Tile, coordinates: &[ExprId]) -> Vec<ExprId> {
        let TileSpace::Iteration { groups, index } = &tile.space else {
            return coordinates.to_vec();
        };
        let mut axes = HashMap::new();
        for (group, &coordinate) in groups.iter().zip(coordinates) {
            let extents = self.axis_extents(group);
            let places = self.delinearize(coordinate, &extents);
            axes.extend(group.iter().copied().zip(places));
        }
        index
            .iter()
            .map(|&part| {
                let read = self.translate_with(part, &axes);
                self.simplify(read)
            })
            .collect()
    }

    /// The values that lie in `[0, bound)` where the element of `tile` at
    /// `coordinates`, which reads the input at `read`, is in the input:
    /// each part of `read` within its axis, and, reindexed, each coordinate
    /// within its group.
    pub(super) fn read_ranges(
        &self,
        tile: &Tile,
        coordinates: &[ExprId],
        read: &[ExprId],
    ) -> Vec<(ExprId, Dim)> {
        let mut ranges = Vec::new();
        if let TileSpace::Iteration { groups, .. } = &tile.space {
            let totals = groups.iter().map(|group| Dim::new(tile_size(&self.axis_extents(group))));
            ranges.extend(coordinates.iter().copied().zip(totals));
        }
        let dims = self.compute.inputs()[tile.tensor].shape.dims();
        ranges.extend(read.iter().copied().zip(dims.iter().copied()));
        ranges
    }

    /// Whether `run` elements of `tile` adjacent along its innermost
    /// coordinate, from a multiple of `run`, are adjacent in one row of the
    /// input, from a multiple of `run`. Reindexed, the innermost group's
    /// innermost axis then steps by one through a whole run, and must step
    /// the input's innermost index part alike and no other part.
    pub(super) fn runs_in_input(&self, tile: &Tile, run: u64) -> bool {
        let compute = self.compute;
        let dims = compute.inputs()[tile.tensor].shape.dims();
        if !dims.last().is_some_and(|innermost| innermost.get().is_multiple_of(run)) {
            return false;
        }
        let TileSpace::Iteration { groups, index } = &tile.space else { return true };
        let innermost = groups.last().and_then(|group| group.last()).copied();
        let Some((axis, var)) = innermost.and_then(|axis| Some((axis, compute.axis_var(axis)?)))
        else {
            return false;
        };
        let Some((&last, outer)) = index.split_last() else { return false };
        compute.iter_vars()[axis].extent.get().is_multiple_of(run)
            && compute.exprs().steps_with(last, var, run)
            && outer.iter().all(|&part| !compute.exprs().vars(part).contains(&var))
    }
}
