//! Applies `fuse`.

use crate::compute::VarName;
use crate::expr::{ExprId, VarId};
use crate::schedule::error::ScheduleError;
use crate::schedule::loop_tree::{LoopId, LoopTree};
use ptxc_core::Dim;

impl LoopTree {
    /// Replaces adjacent loops of one kind, outermost first, by one loop
    /// over their product. Each recovers its index as
    /// `floormod(floordiv(fused, inner extents), extent)`.
    pub(crate) fn fuse(&mut self, loops: &[LoopId]) -> Result<LoopId, ScheduleError> {
        let [first, rest @ ..] = loops else { return Err(ScheduleError::FuseTooFew) };
        if rest.is_empty() {
            return Err(ScheduleError::FuseTooFew);
        }
        let start = self.position(*first)?;
        let kind = self.serial_loop(*first)?.kind;
        for (offset, &id) in rest.iter().enumerate() {
            let fused = self.serial_loop(id)?;
            if fused.kind != kind {
                return Err(ScheduleError::MixedKinds(fused.name.clone()));
            }
            if self.nest.get(start + offset + 1) != Some(&id) {
                return Err(ScheduleError::NotAdjacent(fused.name.clone()));
            }
        }
        let extents = loops
            .iter()
            .map(|&id| i64::try_from(self.loops[id].extent.get()))
            .collect::<Result<Vec<i64>, _>>()
            .map_err(|_| ScheduleError::ExtentOverflow)?;
        let total = extents.iter().try_fold(1_i64, |total, &extent| total.checked_mul(extent));
        let total = total.ok_or(ScheduleError::ExtentOverflow)?;
        let names: Vec<String> = loops.iter().map(|&id| self.loops[id].name.to_string()).collect();
        let fused =
            self.add_loop(VarName::new(names.join("_")), Dim::new(total.unsigned_abs()), kind);
        let _replaced: Vec<LoopId> =
            self.nest.splice(start..start + loops.len(), [fused]).collect();
        let fused_var = self.var(fused);
        let mut inner = total;
        let mut strides = Vec::new();
        for (&id, &extent) in loops.iter().zip(&extents) {
            inner /= extent.max(1);
            strides.push((id, inner));
        }
        self.fold_coordinates(fused, &strides);
        for (&(id, inner), &extent) in strides.iter().zip(&extents) {
            let index = self.unfused_index(fused_var, inner, extent);
            self.replace_var(id, index);
        }
        Ok(fused)
    }

    /// Each reindexed coordinate that reads the loops of `strides` as their
    /// row-major combination, `k·Σ strideᵢ·loopᵢ`, made `k·fused`, as TVM's
    /// iterator maps detect a fusion. Substituting each loop's index would
    /// give the same values within the loops' extents, but a group of unit
    /// extents would simplify to a constant; the fused loop's index keeps
    /// stepping where a split runs it past them, as a tile's coordinate
    /// must for an `mma` to step through the tile.
    fn fold_coordinates(&mut self, fused: LoopId, strides: &[(LoopId, i64)]) {
        let vars: Vec<(VarId, i64)> =
            strides.iter().map(|&(id, stride)| (Self::var_of(id), stride)).collect();
        let fused = Self::var_of(fused);
        let coordinates = self.reindexed.iter_mut().flat_map(|found| &mut found.coordinates);
        for coordinate in coordinates {
            if let Some(folded) = self.exprs.fold_fused(*coordinate, &vars, fused) {
                *coordinate = folded;
            }
        }
    }

    /// `floormod(floordiv(fused, inner), extent)`, one loop's index within a
    /// fused loop.
    fn unfused_index(&mut self, fused: ExprId, inner: i64, extent: i64) -> ExprId {
        let (inner, extent) = (self.exprs.constant(inner), self.exprs.constant(extent));
        let quotient = self.exprs.floor_div(fused, inner);
        self.exprs.floor_mod(quotient, extent)
    }
}
