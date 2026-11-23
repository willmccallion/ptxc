//! Bounds of index expressions from their variables' ranges, by interval
//! arithmetic.

use super::eval::{floor_div, floor_mod};
use super::linear::LinearSum;
use super::{Expr, ExprArena, ExprId, VarId};
use std::collections::HashMap;

/// The integers from `min` to `max`, both included.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Interval {
    min: i64,
    max: i64,
}

impl Interval {
    /// `[min, max]`, or `None` if `min > max`.
    pub const fn new(min: i64, max: i64) -> Option<Self> {
        if min <= max { Some(Self { min, max }) } else { None }
    }

    /// The single value `value`.
    pub const fn point(value: i64) -> Self {
        Self { min: value, max: value }
    }

    /// The smallest value.
    pub const fn min(self) -> i64 {
        self.min
    }

    /// The largest value.
    pub const fn max(self) -> i64 {
        self.max
    }

    fn plus(self, other: Self) -> Option<Self> {
        Some(Self { min: self.min.checked_add(other.min)?, max: self.max.checked_add(other.max)? })
    }

    /// The range of `f` over the corners of `self` and `other`, which bound
    /// it when `f` is monotone in each argument on these intervals.
    fn corners(self, other: Self, f: impl Fn(i64, i64) -> Option<i64>) -> Option<Self> {
        let values = [
            f(self.min, other.min)?,
            f(self.min, other.max)?,
            f(self.max, other.min)?,
            f(self.max, other.max)?,
        ];
        let (min, max) =
            values.iter().fold((i64::MAX, i64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        Some(Self { min, max })
    }

    const fn excludes_zero(self) -> bool {
        self.min > 0 || self.max < 0
    }
}

/// The range of each variable an expression may read; a variable without
/// one is unbounded.
#[derive(Clone, Debug, Default)]
pub struct VarRanges(HashMap<VarId, Interval>);

impl VarRanges {
    /// No known ranges.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that `var` lies in `range`.
    pub fn insert(&mut self, var: VarId, range: Interval) {
        let _previous: Option<Interval> = self.0.insert(var, range);
    }

    /// The range of `var`, if known.
    pub fn get(&self, var: VarId) -> Option<Interval> {
        self.0.get(&var).copied()
    }
}

impl ExprArena {
    /// Bounds of `id`'s value when each variable lies in its range, or
    /// `None` if a variable it reads has no range or a bound overflows.
    pub fn bounds(&self, id: ExprId, ranges: &VarRanges) -> Option<Interval> {
        bounds(self, ranges, id)
    }
}

/// Bounds of `id`'s value over its variables' ranges, or `None` if some
/// variable is unbounded or a bound overflows `i64`.
fn bounds(arena: &ExprArena, ranges: &VarRanges, id: ExprId) -> Option<Interval> {
    let of = |operand| bounds(arena, ranges, operand);
    match arena.get(id) {
        Expr::Var(var) => ranges.get(var),
        Expr::Const(value) => Some(Interval::point(value)),
        Expr::Add(a, b) => of(a)?.plus(of(b)?),
        Expr::Mul(a, b) => of(a)?.corners(of(b)?, i64::checked_mul),
        Expr::FloorDiv(a, b) => {
            let divisor = of(b).filter(|divisor| divisor.excludes_zero())?;
            of(a)?.corners(divisor, floor_div)
        }
        Expr::FloorMod(a, b) => modulo_bounds(of(a), of(b)?),
        Expr::Min(a, b) => {
            let (a, b) = (of(a)?, of(b)?);
            Some(Interval { min: a.min.min(b.min), max: a.max.min(b.max) })
        }
        Expr::Max(a, b) => {
            let (a, b) = (of(a)?, of(b)?);
            Some(Interval { min: a.min.max(b.min), max: a.max.max(b.max) })
        }
    }
}

/// Bounds of `floormod(a, b)`: within one period of a constant divisor the
/// value follows `a`; otherwise it lies between 0 and the divisor, exclusive.
fn modulo_bounds(dividend: Option<Interval>, divisor: Interval) -> Option<Interval> {
    if divisor.min == divisor.max
        && let Some(dividend) = dividend
        && floor_div(dividend.min, divisor.min)? == floor_div(dividend.max, divisor.min)?
    {
        return Some(Interval {
            min: floor_mod(dividend.min, divisor.min)?,
            max: floor_mod(dividend.max, divisor.min)?,
        });
    }
    if divisor.min > 0 {
        Some(Interval { min: 0, max: divisor.max - 1 })
    } else if divisor.max < 0 {
        Some(Interval { min: divisor.min + 1, max: 0 })
    } else {
        None
    }
}

/// Bounds of an affine sum, from its atoms' bounds.
pub(super) fn sum_bounds(
    arena: &ExprArena,
    ranges: &VarRanges,
    sum: &LinearSum,
) -> Option<Interval> {
    let mut total = Interval::point(sum.constant());
    for (atom, coefficient) in sum.terms() {
        let term =
            bounds(arena, ranges, atom)?.corners(Interval::point(coefficient), i64::checked_mul)?;
        total = total.plus(term)?;
    }
    Some(total)
}
