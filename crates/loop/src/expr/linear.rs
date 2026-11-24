//! Affine sums: `Σ cᵢ·tᵢ + k` over atoms `tᵢ`, the sub-expressions that are
//! not sums or constant multiples. Folding a sum through this form combines
//! like terms and puts it in one canonical order.

use super::{Expr, ExprArena, ExprId};
use std::collections::BTreeMap;

/// An affine combination of atoms. Arithmetic on it is checked: `None`
/// means a coefficient overflowed, and the caller keeps the expression as
/// it was.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub(crate) struct LinearSum {
    terms: BTreeMap<ExprId, i64>,
    constant: i64,
}

impl LinearSum {
    pub(super) const fn constant_sum(value: i64) -> Self {
        Self { terms: BTreeMap::new(), constant: value }
    }

    pub(super) fn atom(atom: ExprId) -> Self {
        Self { terms: BTreeMap::from([(atom, 1)]), constant: 0 }
    }

    /// `id` as an affine sum of its atoms.
    pub(crate) fn of(arena: &ExprArena, id: ExprId) -> Option<Self> {
        match arena.get(id) {
            Expr::Const(value) => Some(Self::constant_sum(value)),
            Expr::Add(a, b) => Self::of(arena, a)?.plus(&Self::of(arena, b)?),
            Expr::Mul(a, b) => match (arena.get(a), arena.get(b)) {
                (_, Expr::Const(factor)) => Self::of(arena, a)?.scaled(factor),
                (Expr::Const(factor), _) => Self::of(arena, b)?.scaled(factor),
                _ => Some(Self::atom(id)),
            },
            _ => Some(Self::atom(id)),
        }
    }

    pub(super) const fn constant(&self) -> i64 {
        self.constant
    }

    /// The atoms with nonzero coefficients, in id order.
    pub(crate) fn terms(&self) -> impl Iterator<Item = (ExprId, i64)> + '_ {
        self.terms.iter().map(|(&atom, &coefficient)| (atom, coefficient))
    }

    pub(super) fn coefficient(&self, atom: ExprId) -> i64 {
        self.terms.get(&atom).copied().unwrap_or(0)
    }

    pub(super) fn without(mut self, atom: ExprId) -> Self {
        let _removed: Option<i64> = self.terms.remove(&atom);
        self
    }

    pub(super) fn plus(mut self, other: &Self) -> Option<Self> {
        self.constant = self.constant.checked_add(other.constant)?;
        for (atom, coefficient) in other.terms() {
            let sum = self.coefficient(atom).checked_add(coefficient)?;
            if sum == 0 {
                let _removed: Option<i64> = self.terms.remove(&atom);
            } else {
                let _previous: Option<i64> = self.terms.insert(atom, sum);
            }
        }
        Some(self)
    }

    pub(super) fn scaled(mut self, factor: i64) -> Option<Self> {
        if factor == 0 {
            return Some(Self::default());
        }
        self.constant = self.constant.checked_mul(factor)?;
        for coefficient in self.terms.values_mut() {
            *coefficient = coefficient.checked_mul(factor)?;
        }
        Some(self)
    }

    /// The sum split as `divisor · quotient + rest`: terms whose
    /// coefficients `divisor` divides go to the quotient, and the constant
    /// leaves a remainder in `[0, divisor)` in `rest`. `divisor` is positive.
    pub(super) fn split_divisible(&self, divisor: i64) -> (Self, Self) {
        let mut quotient = Self::constant_sum(self.constant.div_euclid(divisor));
        let mut rest = Self::constant_sum(self.constant.rem_euclid(divisor));
        for (atom, coefficient) in self.terms() {
            if coefficient % divisor == 0 {
                let _new: Option<i64> = quotient.terms.insert(atom, coefficient / divisor);
            } else {
                let _new: Option<i64> = rest.terms.insert(atom, coefficient);
            }
        }
        (quotient, rest)
    }

    /// The canonical expression: terms in atom order, each `atom` or
    /// `atom * coefficient`, then the constant if it is not zero.
    pub(super) fn build(&self, arena: &mut ExprArena) -> ExprId {
        let mut terms = Vec::new();
        for (atom, coefficient) in self.terms() {
            terms.push(if coefficient == 1 {
                atom
            } else {
                let factor = arena.constant(coefficient);
                arena.mul(atom, factor)
            });
        }
        if self.constant != 0 || terms.is_empty() {
            terms.push(arena.constant(self.constant));
        }
        let mut sum = terms[0];
        for &term in &terms[1..] {
            sum = arena.add(sum, term);
        }
        sum
    }
}

impl ExprArena {
    /// Whether every value of `id` is a multiple of `factor`: the constant
    /// of its affine form is, and each term's coefficient or atom is. A
    /// `floormod` is where both its sides are, and a product of two
    /// non-constant factors where either is.
    pub(crate) fn is_multiple_of(&self, id: ExprId, factor: u64) -> bool {
        i64::try_from(factor).is_ok_and(|factor| self.multiple(id, factor))
    }

    fn multiple(&self, id: ExprId, factor: i64) -> bool {
        LinearSum::of(self, id).is_some_and(|sum| {
            sum.constant() % factor == 0
                && sum.terms().all(|(atom, coefficient)| {
                    coefficient % factor == 0 || self.multiple_atom(atom, factor)
                })
        })
    }

    fn multiple_atom(&self, atom: ExprId, factor: i64) -> bool {
        match self.get(atom) {
            Expr::FloorMod(a, b) => self.multiple(a, factor) && self.multiple(b, factor),
            Expr::Mul(a, b) => self.multiple(a, factor) || self.multiple(b, factor),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::expr::{ExprArena, VarId};
    use ptxc_core::Idx;

    #[test]
    fn a_floormod_or_product_is_a_multiple_where_its_parts_make_it_one() {
        let mut exprs = ExprArena::new();
        let [i, j] = [0, 1].map(|var| exprs.var(VarId::new(var)));
        let [four, six, eight] = [4, 6, 8].map(|value| exprs.constant(value));
        let scaled = exprs.mul(i, four);
        let wrapped = exprs.floor_mod(scaled, eight);
        let wrapped_by_six = exprs.floor_mod(scaled, six);
        let times_j = exprs.mul(scaled, j);
        let bare = exprs.mul(i, j);

        let multiples =
            [wrapped, wrapped_by_six, times_j, bare].map(|id| exprs.is_multiple_of(id, 4));

        assert_eq!(multiples, [true, false, true, false]);
    }
}
