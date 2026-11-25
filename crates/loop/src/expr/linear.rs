//! Affine sums: `Σ cᵢ·tᵢ + k` over atoms `tᵢ`, the sub-expressions that are
//! not sums or constant multiples. Folding a sum through this form combines
//! like terms and puts it in one canonical order.

use super::{Expr, ExprArena, ExprId, VarId};
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

    /// Whether `id` is `var` plus terms that read no `var` and whose every
    /// value is a multiple of `factor`, so `factor` adjacent values of
    /// `var` from a multiple of `factor` give as many adjacent values of
    /// `id` from one.
    pub(crate) fn steps_with(&self, id: ExprId, var: VarId, factor: u64) -> bool {
        let Ok(factor) = i64::try_from(factor) else { return false };
        let Some(atom) = self.find(Expr::Var(var)) else { return false };
        LinearSum::of(self, id).is_some_and(|sum| {
            sum.coefficient(atom) == 1
                && sum.constant() % factor == 0
                && sum.terms().filter(|&(term, _)| term != atom).all(|(term, coefficient)| {
                    !self.vars(term).contains(&var)
                        && (coefficient % factor == 0 || self.multiple_atom(term, factor))
                })
        })
    }

    /// `id` with `k·Σ strideᵢ·varᵢ` replaced by `k·fused`, if `id` reads
    /// each of `vars` only in that combination, for one non-zero `k`, every
    /// stride positive: `fused` is the loop fused from `vars`, row-major.
    pub(crate) fn fold_fused(
        &mut self,
        id: ExprId,
        vars: &[(VarId, i64)],
        fused: VarId,
    ) -> Option<ExprId> {
        let sum = LinearSum::of(self, id)?;
        let atoms = vars
            .iter()
            .map(|&(var, stride)| Some((self.find(Expr::Var(var))?, stride)))
            .collect::<Option<Vec<_>>>()?;
        let &(first, first_stride) = atoms.first()?;
        let scale = sum.coefficient(first).checked_div(first_stride).filter(|&scale| scale != 0)?;
        let combined = atoms.iter().all(|&(atom, stride)| {
            stride > 0 && stride.checked_mul(scale) == Some(sum.coefficient(atom))
        });
        let rest = atoms.iter().fold(sum, |rest, &(atom, _)| rest.without(atom));
        let reads_vars = |term: ExprId| {
            self.vars(term).iter().any(|read| vars.iter().any(|&(var, _)| var == *read))
        };
        if !combined || rest.terms().any(|(term, _)| reads_vars(term)) {
            return None;
        }
        let fused = self.var(fused);
        let folded = rest.plus(&LinearSum::atom(fused).scaled(scale)?)?;
        Some(folded.build(self))
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
    use crate::expr::{ExprArena, ExprId, VarId};
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

    #[test]
    fn an_index_steps_with_a_variable_added_once_to_whole_multiples() {
        // `c` steps `c + 8i + 4` and `c + floormod(4j, 8)` four at a time.
        let mut exprs = ExprArena::new();
        let [c, i, j] = [0, 1, 2].map(|var| exprs.var(VarId::new(var)));
        let [four, eight] = [4, 8].map(|value| exprs.constant(value));
        let sum = |exprs: &mut ExprArena, terms: &[ExprId]| {
            terms.iter().copied().reduce(|total, term| exprs.add(total, term)).unwrap()
        };
        let eight_i = exprs.mul(i, eight);
        let four_j = exprs.mul(j, four);
        let wrapped = exprs.floor_mod(four_j, eight);
        let stepping = [sum(&mut exprs, &[c, eight_i, four]), sum(&mut exprs, &[c, wrapped])];

        let steps = stepping.map(|id| exprs.steps_with(id, VarId::new(0), 4));

        assert_eq!(steps, [true, true]);
    }

    #[test]
    fn an_index_does_not_step_with_a_variable_scaled_offset_or_read_twice() {
        let mut exprs = ExprArena::new();
        let [c, i] = [0, 1].map(|var| exprs.var(VarId::new(var)));
        let [two, four] = [2, 4].map(|value| exprs.constant(value));
        let two_c = exprs.mul(c, two);
        let plus_two = exprs.add(c, two);
        let two_i = exprs.mul(i, two);
        let plus_two_i = exprs.add(c, two_i);
        let quarter = exprs.floor_div(c, four);
        let four_quarters = exprs.mul(quarter, four);
        let read_twice = exprs.add(c, four_quarters);

        let steps = [c, two_c, plus_two, plus_two_i, read_twice, i]
            .map(|id| exprs.steps_with(id, VarId::new(0), 4));

        assert_eq!(steps, [true, false, false, false, false, false]);
        assert!(!exprs.steps_with(c, VarId::new(0), u64::MAX));
        assert!(!exprs.steps_with(c, VarId::new(7), 4));
    }

    /// `9a + 3b + c` scaled by `scale`, plus `offset`.
    fn row_major(exprs: &mut ExprArena, scale: i64, offset: ExprId) -> ExprId {
        let [a, b, c] = [0, 1, 2].map(|var| exprs.var(VarId::new(var)));
        let [nine, three, by] = [9, 3, scale].map(|value| exprs.constant(value));
        let (nine_a, three_b) = (exprs.mul(a, nine), exprs.mul(b, three));
        let sum = exprs.add(nine_a, three_b);
        let sum = exprs.add(sum, c);
        let scaled = exprs.mul(sum, by);
        exprs.add(scaled, offset)
    }

    const STRIDES: [(usize, i64); 3] = [(0, 9), (1, 3), (2, 1)];

    fn strides() -> Vec<(VarId, i64)> {
        STRIDES.iter().map(|&(var, stride)| (VarId::new(var), stride)).collect()
    }

    #[test]
    fn a_row_major_combination_of_fused_loops_folds_into_the_fused_loop() {
        let mut exprs = ExprArena::new();
        let other = exprs.var(VarId::new(3));
        let combined = row_major(&mut exprs, 2, other);

        let folded = exprs.fold_fused(combined, &strides(), VarId::new(4)).unwrap();

        let fused = exprs.var(VarId::new(4));
        let two = exprs.constant(2);
        let doubled = exprs.mul(fused, two);
        let expected = [exprs.add(other, doubled), exprs.add(doubled, other)];
        assert!(expected.contains(&folded), "{:?}", exprs.get(folded));
    }

    #[test]
    fn a_sum_reading_the_loops_otherwise_does_not_fold() {
        let mut exprs = ExprArena::new();
        let zero = exprs.constant(0);
        let a = exprs.var(VarId::new(0));
        let [nine, four] = [9, 4].map(|value| exprs.constant(value));
        let skewed = {
            let combined = row_major(&mut exprs, 1, zero);
            let extra = exprs.mul(a, nine);
            exprs.add(combined, extra)
        };
        let divided = {
            let quarter = exprs.floor_div(a, four);
            row_major(&mut exprs, 1, quarter)
        };
        let unit = row_major(&mut exprs, 1, zero);

        let folds = [skewed, divided].map(|id| exprs.fold_fused(id, &strides(), VarId::new(4)));
        let unread = exprs.fold_fused(unit, &[(VarId::new(9), 1)], VarId::new(4));
        let zero_stride = exprs.fold_fused(unit, &[(VarId::new(0), 0)], VarId::new(4));
        let _unread = exprs.var(VarId::new(5));
        let inner_zero =
            [(VarId::new(0), 9), (VarId::new(1), 3), (VarId::new(2), 1), (VarId::new(5), 0)];
        let zero_inner = exprs.fold_fused(unit, &inner_zero, VarId::new(4));

        assert_eq!(folds, [None, None]);
        assert_eq!([unread, zero_stride, zero_inner], [None, None, None]);
    }
}
