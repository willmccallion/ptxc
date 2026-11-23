//! Applies `split`.

use crate::compute::VarName;
use crate::expr::ExprId;
use crate::schedule::error::ScheduleError;
use crate::schedule::loop_tree::{Guard, LoopId, LoopTree};
use crate::schedule::primitive::SplitFactor;
use ptxc_core::Dim;

impl LoopTree {
    /// Replaces `target` by an outer loop and one loop per factor, whose
    /// indices recombine as `outer * (f1 * ... * fn) + ... + inner`. When
    /// the factors' product does not divide the extent, the outer loop
    /// rounds up and a guard keeps the recombined index below the extent.
    pub(crate) fn split(
        &mut self,
        target: LoopId,
        factors: &[SplitFactor],
    ) -> Result<Vec<LoopId>, ScheduleError> {
        let original = self.serial_loop(target)?.clone();
        if factors.is_empty() {
            return Err(ScheduleError::EmptySplit);
        }
        let position = self.position(target)?;
        let strides = strides(factors)?;
        let outer = original.extent.get().div_ceil(strides[0].unsigned_abs());
        let extents = std::iter::once(outer).chain(factors.iter().map(|factor| factor.get()));
        let parts: Vec<LoopId> = extents
            .enumerate()
            .map(|(part, extent)| {
                let name = VarName::new(format!("{}.{part}", original.name));
                self.add_loop(name, Dim::new(extent), original.kind)
            })
            .collect();
        let _replaced: Vec<LoopId> =
            self.nest.splice(position..=position, parts.iter().copied()).collect();
        let index = self.recombined(&parts, &strides);
        self.replace_var(target, index);
        if original.extent.get() % strides[0].unsigned_abs() != 0 {
            let index = self.simplified(index);
            self.guards.push(Guard { index, extent: original.extent });
        }
        Ok(parts)
    }

    /// `sum(part * stride)` over the parts.
    fn recombined(&mut self, parts: &[LoopId], strides: &[i64]) -> ExprId {
        let mut index = self.exprs.constant(0);
        for (&part, &stride) in parts.iter().zip(strides).rev() {
            let var = self.var(part);
            let scale = self.exprs.constant(stride);
            let term = self.exprs.mul(var, scale);
            index = self.exprs.add(term, index);
        }
        index
    }
}

/// Each part's stride, outermost first: the product of the factors inside
/// it, so the outer part's stride is the product of them all. Strides must
/// fit an index constant.
fn strides(factors: &[SplitFactor]) -> Result<Vec<i64>, ScheduleError> {
    let mut strides = vec![1_i64];
    for factor in factors.iter().rev() {
        let factor = i64::try_from(factor.get()).map_err(|_| ScheduleError::ExtentOverflow)?;
        let next = strides[0].checked_mul(factor).ok_or(ScheduleError::ExtentOverflow)?;
        strides.insert(0, next);
    }
    Ok(strides)
}
