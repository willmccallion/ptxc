//! What the enclosing loops and conditions establish about index values.

use crate::expr::{Expr, ExprArena, ExprId, Interval, VarId, VarRanges};
use crate::stmt::Condition;
use std::collections::HashMap;

/// Each bound variable's range, and the cap each enclosing condition puts
/// on its whole index expression.
#[derive(Clone, Debug, Default)]
pub(super) struct Facts {
    ranges: VarRanges,
    /// The largest value of `e` under a condition `e < b` with constant `b`.
    /// The arena shares equal expressions, so an access at `e` finds it.
    caps: HashMap<ExprId, i64>,
}

impl Facts {
    /// `var` takes only the values in `range`.
    pub(super) fn insert_range(&mut self, var: VarId, range: Interval) {
        self.ranges.insert(var, range);
    }

    /// Records that `condition` holds, as a cap on its index, and as a
    /// tighter range for `v` when the index is `v + c` for a constant `c`.
    pub(super) fn refine(&mut self, exprs: &ExprArena, condition: Condition) {
        let Expr::Const(bound) = exprs.get(condition.bound) else { return };
        let cap = self.caps.entry(condition.index).or_insert(i64::MAX);
        *cap = (*cap).min(bound.saturating_sub(1));
        let (var, offset) = match exprs.get(condition.index) {
            Expr::Var(var) => (var, 0),
            Expr::Add(a, b) => match (exprs.get(a), exprs.get(b)) {
                (Expr::Var(var), Expr::Const(offset)) | (Expr::Const(offset), Expr::Var(var)) => {
                    (var, offset)
                }
                _ => return,
            },
            _ => return,
        };
        let Some(range) = self.ranges.get(var) else { return };
        let below = bound.saturating_sub(offset);
        if let Some(tighter) = Interval::new(range.min(), range.max().min(below - 1)) {
            self.ranges.insert(var, tighter);
        }
    }

    /// Bounds of `index`'s value over the variables' ranges, within any cap
    /// a condition puts on `index` itself.
    pub(super) fn bounds(&self, exprs: &ExprArena, index: ExprId) -> Option<Interval> {
        let bounds = exprs.bounds(index, &self.ranges)?;
        match self.caps.get(&index) {
            Some(&cap) => Interval::new(bounds.min(), bounds.max().min(cap)),
            None => Some(bounds),
        }
    }
}
