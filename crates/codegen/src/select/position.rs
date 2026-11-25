//! An element's position in its buffer, split as its address puts it
//! together. The index is taken apart into terms, constant factors
//! distributed over sums, and the constant becomes the address's immediate
//! offset. In a swizzled buffer, terms that are whole multiples of the
//! swizzle's period, such as a pipeline slot's or a warp's tile origin, are
//! added after the swizzle, which moves them unchanged, and the constant is
//! folded in after it (D67); only the rest is swizzled. The parts are added
//! outermost loop first, as LLVM's Reassociate ranks operands, so the part
//! of an address a loop does not change is a sum of its own.

use super::{CodegenError, Selector};
use crate::ast::{B64, BinaryOp, PtxType, Reg, Signedness, Value};
use ptxc_loop::{BufferId, Expr, ExprArena, ExprId, KernelVar};
use std::collections::BTreeMap;

/// An element's position in its buffer, in elements.
#[derive(Clone, Debug)]
pub(super) struct Position {
    /// The part the buffer's swizzle places, if it has one and any.
    pub(super) placed: Option<Ranked>,
    /// The parts the buffer's layout moves unchanged, one a rank,
    /// outermost first.
    pub(super) shifted: Vec<Ranked>,
    /// The constant, folded in after the placed part.
    pub(super) constant: i64,
}

/// Part of a position, and its rank: the depth of the innermost loop whose
/// variable it reads, 0 for none.
#[derive(Clone, Copy, Debug)]
pub(super) struct Ranked {
    pub(super) rank: u32,
    pub(super) value: Reg<B64>,
}

/// A term of an index: an expression times a constant.
#[derive(Clone, Copy, Debug)]
struct Term {
    expr: ExprId,
    coefficient: i64,
}

/// An index's terms by how its address treats them.
struct Terms {
    placed: Vec<Term>,
    shifted: Vec<Term>,
    constant: i64,
}

impl Selector<'_> {
    /// Where element `index` of `buffer` is.
    pub(super) fn position(
        &mut self,
        buffer: BufferId,
        index: ExprId,
    ) -> Result<Position, CodegenError> {
        self.position_with(buffer, index, None)
    }

    /// Where element `index` of `buffer` is, plus `within`, which the
    /// buffer's layout places with the index's placed terms.
    pub(super) fn position_with(
        &mut self,
        buffer: BufferId,
        index: ExprId,
        within: Option<Ranked>,
    ) -> Result<Position, CodegenError> {
        let terms = self.terms(buffer, index)?;
        let placed = self.placed_sum(within, &terms.placed)?;
        let mut shifted = Vec::new();
        for group in self.by_rank(&terms.shifted).values() {
            shifted.extend(self.ranked_sum(group)?);
        }
        Ok(Position { placed, shifted, constant: terms.constant })
    }

    /// `within` plus the sum of `placed`, if either is given.
    fn placed_sum(
        &mut self,
        within: Option<Ranked>,
        placed: &[Term],
    ) -> Result<Option<Ranked>, CodegenError> {
        let sum = self.ranked_sum(placed)?;
        Ok(match (within, sum) {
            (Some(within), Some(sum)) => {
                let value = self.added(within.value, sum.value);
                Some(Ranked { rank: within.rank.max(sum.rank), value })
            }
            (within, sum) => within.or(sum),
        })
    }

    /// `index`'s terms: in a plain buffer all moved unchanged, in a
    /// swizzled one only those that are whole multiples of its period.
    fn terms(&self, buffer: BufferId, index: ExprId) -> Result<Terms, CodegenError> {
        let (linear, constant) = self.linear_terms(index)?;
        let Some(swizzle) = self.swizzle_of(buffer) else {
            return Ok(Terms { placed: Vec::new(), shifted: linear, constant });
        };
        let element = self.kernel.buffers()[buffer].dtype.element_size().get();
        let mut terms = Terms { placed: Vec::new(), shifted: Vec::new(), constant };
        for term in linear {
            let twos = self.twos(term.expr, None);
            if moves_unchanged((twos, term.coefficient), element, swizzle.period_bytes()) {
                terms.shifted.push(term);
            } else {
                terms.placed.push(term);
            }
        }
        Ok(terms)
    }

    /// Index expression `id`, a sum or a product by a constant: its terms
    /// added outermost rank first, then its constant.
    pub(super) fn linear_index(&mut self, id: ExprId) -> Result<Reg<B64>, CodegenError> {
        let (terms, constant) = self.linear_terms(id)?;
        let sum = self.ranked_sum(&terms)?;
        let constant_value = Value::<B64>::int(constant);
        Ok(match sum {
            Some(sum) if constant == 0 => sum.value,
            Some(sum) => {
                self.index_op(BinaryOp::Add, Signedness::Signed, sum.value.into(), constant_value)
            }
            None => self.builder.mov(PtxType::S64, constant_value),
        })
    }

    /// `index` as a sum of terms and a constant, constant factors
    /// distributed over sums so each term's factors of two show.
    fn linear_terms(&self, index: ExprId) -> Result<(Vec<Term>, i64), CodegenError> {
        let exprs = self.kernel.exprs();
        let scaled = |coefficient: i64, by: i64| coefficient.checked_mul(by);
        let (mut terms, mut constant) = (Vec::new(), 0_i64);
        let mut pending = vec![Term { expr: index, coefficient: 1 }];
        while let Some(term) = pending.pop() {
            let Term { expr, coefficient } = term;
            match exprs.get(expr) {
                Expr::Add(a, b) => pending.extend([b, a].map(|expr| Term { expr, ..term })),
                Expr::Mul(a, b) => match constant_factor(exprs, a, b) {
                    Some((factor, by)) => {
                        let coefficient = scaled(coefficient, by).ok_or(CodegenError::TooLarge)?;
                        pending.push(Term { expr: factor, coefficient });
                    }
                    None => terms.push(term),
                },
                Expr::Const(value) => {
                    let value = scaled(coefficient, value).ok_or(CodegenError::TooLarge)?;
                    constant = constant.checked_add(value).ok_or(CodegenError::TooLarge)?;
                }
                _ => terms.push(term),
            }
        }
        Ok((terms, constant))
    }

    /// `terms` grouped by rank, each group in the order given.
    fn by_rank(&self, terms: &[Term]) -> BTreeMap<u32, Vec<Term>> {
        let mut groups: BTreeMap<u32, Vec<Term>> = BTreeMap::new();
        for &term in terms {
            groups.entry(self.rank(term.expr)).or_default().push(term);
        }
        groups
    }

    /// How deep in loops `expr` varies: the depth of the innermost serial
    /// loop whose variable it reads, 0 for none, and past every loop where
    /// it reads one not yet bound or loaded from memory.
    fn rank(&self, expr: ExprId) -> u32 {
        match self.kernel.exprs().get(expr) {
            Expr::Var(var) => match self.kernel.vars()[var] {
                KernelVar::Thread { .. } => 0,
                KernelVar::Loop(_) => self.loop_depths.get(&var).copied().unwrap_or(u32::MAX),
                KernelVar::Loaded(_) => u32::MAX,
            },
            Expr::Const(_) => 0,
            Expr::Add(a, b)
            | Expr::Mul(a, b)
            | Expr::Min(a, b)
            | Expr::Max(a, b)
            | Expr::FloorDiv(a, b)
            | Expr::FloorMod(a, b) => self.rank(a).max(self.rank(b)),
        }
    }

    /// The sum of `terms`, outermost rank first, and its rank, if there are
    /// any.
    fn ranked_sum(&mut self, terms: &[Term]) -> Result<Option<Ranked>, CodegenError> {
        let mut sum: Option<Ranked> = None;
        for group in self.by_rank(terms).into_values() {
            for term in group {
                let scaled = self.scaled_index(term)?;
                let value = sum.map_or(scaled, |sum| self.added(sum.value, scaled));
                sum = Some(Ranked { rank: self.rank(term.expr), value });
            }
        }
        Ok(sum)
    }

    /// `term`'s expression times its coefficient.
    fn scaled_index(&mut self, Term { expr, coefficient }: Term) -> Result<Reg<B64>, CodegenError> {
        let value = self.index(expr)?;
        if coefficient == 1 {
            return Ok(value);
        }
        let by = Value::<B64>::int(coefficient);
        Ok(self.index_op(BinaryOp::Mul, Signedness::Signed, value.into(), by))
    }

    /// `a + b`.
    pub(super) fn added(&mut self, a: Reg<B64>, b: Reg<B64>) -> Reg<B64> {
        self.index_op(BinaryOp::Add, Signedness::Signed, a.into(), b.into())
    }
}

/// `a * b` in `exprs` as an expression times a constant, if either is one.
pub(super) fn constant_factor(exprs: &ExprArena, a: ExprId, b: ExprId) -> Option<(ExprId, i64)> {
    match (exprs.get(a), exprs.get(b)) {
        (_, Expr::Const(by)) => Some((a, by)),
        (Expr::Const(by), _) => Some((b, by)),
        _ => None,
    }
}

/// Whether a swizzle of `period_bytes` moves a term unchanged: a term with
/// `twos` factors of two times `coefficient`, of elements of
/// `element_bytes`, whose every value is then a whole multiple of the
/// period.
const fn moves_unchanged(
    (twos, coefficient): (u32, i64),
    element_bytes: u64,
    period_bytes: u64,
) -> bool {
    twos + coefficient.trailing_zeros() + element_bytes.trailing_zeros()
        >= period_bytes.trailing_zeros()
}

#[cfg(test)]
mod tests {
    use super::{constant_factor, moves_unchanged};
    use ptxc_core::Idx;
    use ptxc_loop::{ExprArena, VarId};

    #[test]
    fn a_product_by_a_constant_on_either_side_is_that_constant_times_the_rest() {
        let mut exprs = ExprArena::new();
        let (x, four) = (exprs.var(VarId::new(0)), exprs.constant(4));
        let [right, left, both] = [(x, four), (four, x), (x, x)].map(|(a, b)| exprs.mul(a, b));
        let factor = |product| match exprs.get(product) {
            ptxc_loop::Expr::Mul(a, b) => constant_factor(&exprs, a, b),
            _ => None,
        };

        assert_eq!([right, left, both].map(factor), [Some((x, 4)), Some((x, 4)), None]);
    }

    #[test]
    fn a_term_moves_unchanged_from_a_whole_period_of_bytes_up() {
        // f16 elements under a 1 KiB period: 512 elements are a period.
        let period = 1024;

        let moves = |twos, coefficient| moves_unchanged((twos, coefficient), 2, period);

        assert!(moves(0, 512) && moves(9, 1) && moves(4, 32) && moves(0, -512));
        assert!(!moves(0, 256) && !moves(8, 1) && !moves(3, 32));
    }
}
