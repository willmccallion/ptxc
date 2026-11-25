//! Index simplification: affine folding, and floor division and modulo
//! rules that use the variables' ranges, after TVM's arith rewrite rules.

use super::eval::{floor_div, floor_mod};
use super::interval::{VarRanges, sum_bounds};
use super::linear::LinearSum;
use super::{Expr, ExprArena, ExprId};
use std::collections::HashMap;

/// Simplifies expressions given the ranges of their variables.
///
/// The result has the same value as the input for every assignment within
/// the ranges under which the input evaluates. Results are memoised, so a
/// simplifier is used with one arena and one set of ranges.
#[derive(Debug)]
pub struct Simplifier<'r> {
    ranges: &'r VarRanges,
    memo: HashMap<ExprId, ExprId>,
}

/// Which extreme a `Min` or `Max` keeps.
#[derive(Clone, Copy)]
enum Extreme {
    Min,
    Max,
}

impl<'r> Simplifier<'r> {
    /// A simplifier for variables in `ranges`.
    pub fn new(ranges: &'r VarRanges) -> Self {
        Self { ranges, memo: HashMap::new() }
    }

    /// The simplest equal expression to `id` this simplifier finds.
    pub fn simplify(&mut self, arena: &mut ExprArena, id: ExprId) -> ExprId {
        if let Some(&done) = self.memo.get(&id) {
            return done;
        }
        let simplified = self.simplify_node(arena, arena.get(id));
        let _first: Option<ExprId> = self.memo.insert(id, simplified);
        let _settled: Option<ExprId> = self.memo.insert(simplified, simplified);
        simplified
    }

    fn simplify_node(&mut self, arena: &mut ExprArena, expr: Expr) -> ExprId {
        match expr {
            Expr::Var(_) | Expr::Const(_) => arena.intern(expr),
            Expr::Add(a, b) => {
                let (a, b) = (self.simplify(arena, a), self.simplify(arena, b));
                let sum = arena.add(a, b);
                affine(arena, sum)
            }
            Expr::Mul(a, b) => {
                let (a, b) = (self.simplify(arena, a), self.simplify(arena, b));
                let product = arena.mul(a.min(b), a.max(b));
                affine(arena, product)
            }
            Expr::FloorDiv(a, b) => {
                let (a, b) = (self.simplify(arena, a), self.simplify(arena, b));
                self.floor_div(arena, a, b)
            }
            Expr::FloorMod(a, b) => {
                let (a, b) = (self.simplify(arena, a), self.simplify(arena, b));
                self.floor_mod(arena, a, b)
            }
            Expr::Min(a, b) => {
                let (a, b) = (self.simplify(arena, a), self.simplify(arena, b));
                self.extreme(arena, Extreme::Min, a, b)
            }
            Expr::Max(a, b) => {
                let (a, b) = (self.simplify(arena, a), self.simplify(arena, b));
                self.extreme(arena, Extreme::Max, a, b)
            }
        }
    }

    /// `floordiv(a, b)` for simplified operands. With a positive constant
    /// divisor `c`, `floordiv(c·q + r, c) = q + floordiv(r, c)`, and the
    /// last term is a constant when `r`'s bounds lie within one multiple of
    /// `c`.
    fn floor_div(&mut self, arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
        let Some(divisor) = positive_constant(arena, b) else {
            return fold(arena, a, b, floor_div).unwrap_or_else(|| arena.floor_div(a, b));
        };
        if let Some(folded) = fold(arena, a, b, floor_div) {
            return folded;
        }
        let Some(sum) = LinearSum::of(arena, a) else { return arena.floor_div(a, b) };
        let (quotient, rest) = sum.split_divisible(divisor);
        let tail = match sum_bounds(arena, self.ranges, &rest)
            .map(|range| (floor_div(range.min(), divisor), floor_div(range.max(), divisor)))
        {
            Some((Some(low), Some(high))) if low == high => LinearSum::constant_sum(low),
            _ => {
                let rest = rest.build(arena);
                LinearSum::atom(self.nested_floor_div(arena, rest, divisor))
            }
        };
        match quotient.plus(&tail) {
            Some(total) => build_fused(arena, total),
            None => arena.floor_div(a, b),
        }
    }

    /// `floordiv(r, c)`, merging `floordiv(floordiv(x, d), c)` into
    /// `floordiv(x, d·c)` for positive `d` and `c`.
    fn nested_floor_div(&mut self, arena: &mut ExprArena, rest: ExprId, divisor: i64) -> ExprId {
        if let Expr::FloorDiv(x, inner) = arena.get(rest)
            && let Some(product) =
                positive_constant(arena, inner).and_then(|d| d.checked_mul(divisor))
        {
            let combined = arena.constant(product);
            return self.floor_div(arena, x, combined);
        }
        let divisor = arena.constant(divisor);
        arena.floor_div(rest, divisor)
    }

    /// `floormod(a, b)` for simplified operands. With a positive constant
    /// divisor `c`, `floormod(c·q + r, c) = floormod(r, c)`, which is `r`
    /// shifted by a multiple of `c` when `r`'s bounds lie within one period.
    fn floor_mod(&mut self, arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
        let Some(divisor) = positive_constant(arena, b) else {
            return fold(arena, a, b, floor_mod).unwrap_or_else(|| arena.floor_mod(a, b));
        };
        if let Some(folded) = fold(arena, a, b, floor_mod) {
            return folded;
        }
        let Some(sum) = LinearSum::of(arena, a) else { return arena.floor_mod(a, b) };
        let (_, rest) = sum.split_divisible(divisor);
        let period = sum_bounds(arena, self.ranges, &rest).and_then(|range| {
            let low = floor_div(range.min(), divisor)?;
            (low == floor_div(range.max(), divisor)?).then_some(low)
        });
        if let Some(shifted) = period
            .and_then(|period| period.checked_mul(divisor))
            .and_then(|offset| rest.clone().plus(&LinearSum::constant_sum(offset.checked_neg()?)))
        {
            return build_fused(arena, shifted);
        }
        let rest = rest.build(arena);
        self.nested_floor_mod(arena, rest, divisor)
    }

    /// `floormod(r, c)`, merging `floormod(floormod(x, d), c)` into
    /// `floormod(x, c)` when `c` divides `d`.
    fn nested_floor_mod(&mut self, arena: &mut ExprArena, rest: ExprId, divisor: i64) -> ExprId {
        if let Expr::FloorMod(x, inner) = arena.get(rest)
            && positive_constant(arena, inner).is_some_and(|d| d % divisor == 0)
        {
            let divisor = arena.constant(divisor);
            return self.floor_mod(arena, x, divisor);
        }
        let divisor = arena.constant(divisor);
        arena.floor_mod(rest, divisor)
    }

    /// `min(a, b)` or `max(a, b)` for simplified operands: folded, decided
    /// by the bounds of `a - b`, or kept with its operands in id order.
    fn extreme(&self, arena: &mut ExprArena, which: Extreme, a: ExprId, b: ExprId) -> ExprId {
        if a == b {
            return a;
        }
        let difference = LinearSum::of(arena, a)
            .zip(LinearSum::of(arena, b).and_then(|b| b.scaled(-1)))
            .and_then(|(a, minus_b)| a.plus(&minus_b))
            .and_then(|difference| sum_bounds(arena, self.ranges, &difference));
        match (which, difference) {
            (Extreme::Min, Some(range)) if range.max() <= 0 => a,
            (Extreme::Min, Some(range)) if range.min() >= 0 => b,
            (Extreme::Max, Some(range)) if range.min() >= 0 => a,
            (Extreme::Max, Some(range)) if range.max() <= 0 => b,
            (Extreme::Min, _) => arena.min(a.min(b), a.max(b)),
            (Extreme::Max, _) => arena.max(a.min(b), a.max(b)),
        }
    }
}

/// The constant `id` is, if it is a positive constant.
fn positive_constant(arena: &ExprArena, id: ExprId) -> Option<i64> {
    match arena.get(id) {
        Expr::Const(value) if value > 0 => Some(value),
        _ => None,
    }
}

/// `f(a, b)` as a constant, when both are constants and `f` is defined.
fn fold(
    arena: &mut ExprArena,
    a: ExprId,
    b: ExprId,
    f: fn(i64, i64) -> Option<i64>,
) -> Option<ExprId> {
    let (Expr::Const(a), Expr::Const(b)) = (arena.get(a), arena.get(b)) else { return None };
    f(a, b).map(|value| arena.constant(value))
}

/// `id`, a sum or product of simplified operands, in canonical affine form.
fn affine(arena: &mut ExprArena, id: ExprId) -> ExprId {
    LinearSum::of(arena, id).map_or(id, |sum| build_fused(arena, sum))
}

/// The canonical expression for `sum`, after replacing each
/// `k·f·floordiv(x, f) + k·floormod(x, f)` with `k·x`.
fn build_fused(arena: &mut ExprArena, sum: LinearSum) -> ExprId {
    let mut sum = sum;
    while let Some(fused) = fuse_once(arena, &sum) {
        sum = fused;
    }
    sum.build(arena)
}

/// `sum` with one `floordiv`/`floormod` pair recombined, if it has one.
fn fuse_once(arena: &ExprArena, sum: &LinearSum) -> Option<LinearSum> {
    sum.terms().find_map(|(atom, coefficient)| {
        let Expr::FloorMod(x, divisor) = arena.get(atom) else { return None };
        // The identity needs a divisor; with 0 both terms are undefined.
        let Expr::Const(factor) = arena.get(divisor) else { return None };
        if factor == 0 {
            return None;
        }
        let quotient =
            arena.find(Expr::FloorDiv(x, divisor)).or_else(|| merged_quotient(arena, x, factor))?;
        if sum.coefficient(quotient) != coefficient.checked_mul(factor)? {
            return None;
        }
        let rest = sum.clone().without(atom).without(quotient);
        rest.plus(&LinearSum::of(arena, x)?.scaled(coefficient)?)
    })
}

/// `floordiv(x, factor)` where `x` is `floordiv(y, d)`, in the form
/// `nested_floor_div` merges it into, `floordiv(y, d·factor)`, if the arena
/// holds it. The merge needs positive divisors. Three or more fused axes
/// recombine through it: the middle axis's `floormod(floordiv(y, d), f)`
/// pairs with the outer axis's `floordiv(y, d·f)`.
fn merged_quotient(arena: &ExprArena, x: ExprId, factor: i64) -> Option<ExprId> {
    let Expr::FloorDiv(y, inner) = arena.get(x) else { return None };
    let product = positive_constant(arena, inner)?.checked_mul(factor).filter(|_| factor > 0)?;
    let divisor = arena.find(Expr::Const(product))?;
    arena.find(Expr::FloorDiv(y, divisor))
}
