//! Each simplifier rule, on the index expressions schedules produce.

use ptxc_core::Idx;
use ptxc_loop::{Expr, ExprArena, ExprId, Interval, Simplifier, VarId, VarRanges};

fn var(index: usize) -> VarId {
    VarId::new(index)
}

/// An arena with variables `x` (unbounded), `outer` in `[0, 9]` and
/// `inner` in `[0, 3]`, as `split(i, 4)` makes them.
struct Fixture {
    arena: ExprArena,
    ranges: VarRanges,
    x: ExprId,
    outer: ExprId,
    inner: ExprId,
}

fn fixture() -> Fixture {
    let mut arena = ExprArena::new();
    let mut ranges = VarRanges::new();
    ranges.insert(var(1), Interval::new(0, 9).unwrap());
    ranges.insert(var(2), Interval::new(0, 3).unwrap());
    let (x, outer, inner) = (arena.var(var(0)), arena.var(var(1)), arena.var(var(2)));
    Fixture { arena, ranges, x, outer, inner }
}

impl Fixture {
    fn c(&mut self, value: i64) -> ExprId {
        self.arena.constant(value)
    }

    fn simplify(&mut self, id: ExprId) -> ExprId {
        Simplifier::new(&self.ranges).simplify(&mut self.arena, id)
    }

    /// `outer * 4 + inner`, the index a split by 4 reconstructs.
    fn split_index(&mut self) -> ExprId {
        let four = self.c(4);
        let scaled = self.arena.mul(self.outer, four);
        self.arena.add(scaled, self.inner)
    }
}

#[test]
fn a_split_index_divides_back_into_its_loops() {
    let mut f = fixture();
    let index = f.split_index();
    let four = f.c(4);
    let quotient = f.arena.floor_div(index, four);
    let remainder = f.arena.floor_mod(index, four);

    assert_eq!(f.simplify(quotient), f.outer);
    assert_eq!(f.simplify(remainder), f.inner);
}

#[test]
fn a_constant_offset_of_whole_periods_moves_to_the_quotient() {
    let mut f = fixture();
    let index = f.split_index();
    let (four, eight) = (f.c(4), f.c(8));
    let shifted = f.arena.add(index, eight);
    let quotient = f.arena.floor_div(shifted, four);
    let remainder = f.arena.floor_mod(shifted, four);

    let two = f.c(2);
    let expected_quotient = f.arena.add(f.outer, two);
    assert_eq!(f.simplify(quotient), expected_quotient);
    assert_eq!(f.simplify(remainder), f.inner);
}

#[test]
fn a_remainder_that_crosses_a_period_stays_divided() {
    let mut f = fixture();
    let index = f.split_index();
    let (four, six, two, one) = (f.c(4), f.c(6), f.c(2), f.c(1));
    let shifted = f.arena.add(index, six);
    let quotient = f.arena.floor_div(shifted, four);
    let remainder = f.arena.floor_mod(shifted, four);

    // `inner + 2` spans [2, 5], so it crosses 4.
    let inner_plus_two = f.arena.add(f.inner, two);
    let carry = f.arena.floor_div(inner_plus_two, four);
    let outer_plus_carry = f.arena.add(f.outer, carry);
    let expected_quotient = f.arena.add(outer_plus_carry, one);
    let expected_remainder = f.arena.floor_mod(inner_plus_two, four);
    assert_eq!(f.simplify(quotient), expected_quotient);
    assert_eq!(f.simplify(remainder), expected_remainder);
}

#[test]
fn a_quotient_and_remainder_fuse_back_into_the_index() {
    let mut f = fixture();
    let eight = f.c(8);
    let quotient = f.arena.floor_div(f.x, eight);
    let scaled = f.arena.mul(quotient, eight);
    let remainder = f.arena.floor_mod(f.x, eight);
    let fused = f.arena.add(scaled, remainder);

    assert_eq!(f.simplify(fused), f.x);
}

#[test]
fn like_terms_combine_and_constants_cancel() {
    let mut f = fixture();
    let (two, three, minus_three) = (f.c(2), f.c(3), f.c(-3));
    let left = f.arena.add(f.x, three);
    let doubled = f.arena.mul(two, f.x);
    let right = f.arena.add(doubled, minus_three);
    let sum = f.arena.add(left, right);

    let expected = f.arena.mul(f.x, three);
    assert_eq!(f.simplify(sum), expected);
}

#[test]
fn nested_divisions_and_moduli_merge() {
    let mut f = fixture();
    let (four, eight, thirty_two) = (f.c(4), f.c(8), f.c(32));
    let inner_div = f.arena.floor_div(f.x, four);
    let outer_div = f.arena.floor_div(inner_div, eight);
    let inner_mod = f.arena.floor_mod(f.x, eight);
    let outer_mod = f.arena.floor_mod(inner_mod, four);

    let expected_div = f.arena.floor_div(f.x, thirty_two);
    let expected_mod = f.arena.floor_mod(f.x, four);
    assert_eq!(f.simplify(outer_div), expected_div);
    assert_eq!(f.simplify(outer_mod), expected_mod);
}

#[test]
fn a_remainder_divided_by_its_modulus_is_zero() {
    let mut f = fixture();
    let four = f.c(4);
    let remainder = f.arena.floor_mod(f.x, four);
    let quotient = f.arena.floor_div(remainder, four);

    let zero = f.c(0);
    assert_eq!(f.simplify(quotient), zero);
}

#[test]
fn min_and_max_are_decided_by_ranges() {
    let mut f = fixture();
    let (one, four) = (f.c(1), f.c(4));
    let below = f.arena.min(f.inner, four);
    let next = f.arena.add(f.x, one);
    let larger = f.arena.max(f.x, next);
    let undecided = f.arena.min(f.x, f.inner);

    assert_eq!(f.simplify(below), f.inner);
    assert_eq!(f.simplify(larger), next);
    assert_eq!(f.simplify(undecided), f.arena.min(f.x, f.inner));
}

#[test]
fn division_by_zero_is_left_for_evaluation_to_reject() {
    let mut f = fixture();
    let zero = f.c(0);
    let quotient = f.arena.floor_div(f.x, zero);
    let remainder = f.arena.floor_mod(f.x, zero);

    assert_eq!(f.simplify(quotient), quotient);
    assert_eq!(f.simplify(remainder), remainder);
}

#[test]
fn a_negative_divisor_folds_constants_but_keeps_variables() {
    let mut f = fixture();
    let (seven, minus_two) = (f.c(7), f.c(-2));
    let constant = f.arena.floor_div(seven, minus_two);
    let variable = f.arena.floor_mod(f.x, minus_two);

    let expected = f.c(-4);
    assert_eq!(f.simplify(constant), expected);
    assert_eq!(f.simplify(variable), variable);
}

/// A fixture variable with `range`, numbered after the fixture's own.
fn ranged(f: &mut Fixture, index: usize, low: i64, high: i64) -> ExprId {
    f.ranges.insert(var(index), Interval::new(low, high).unwrap());
    f.arena.var(var(index))
}

#[test]
fn division_by_a_range_of_one_sign_is_bounded() {
    let mut f = fixture();
    let positive = ranged(&mut f, 3, 1, 4);
    let negative = ranged(&mut f, 4, -4, -1);
    let (three, zero) = (f.c(3), f.c(0));
    let by_positive = f.arena.floor_div(f.inner, positive);
    let by_negative = f.arena.floor_div(f.inner, negative);
    let at_most_three = f.arena.min(by_positive, three);
    let at_most_zero = f.arena.min(by_negative, zero);

    assert_eq!(f.simplify(at_most_three), by_positive);
    assert_eq!(f.simplify(at_most_zero), by_negative);
}

#[test]
fn division_by_a_range_that_crosses_zero_is_unbounded() {
    let mut f = fixture();
    let crossing = ranged(&mut f, 3, -2, 3);
    let one = f.c(1);
    let quotient = f.arena.floor_div(f.inner, crossing);
    let at_most_one = f.arena.min(quotient, one);

    // With a divisor of 1 the quotient reaches 3, so the min must stay.
    let simplified = f.simplify(at_most_one);
    let Expr::Min(a, b) = f.arena.get(simplified) else { panic!("min decided") };
    assert_eq!([a.min(b), a.max(b)], [quotient.min(one), quotient.max(one)]);
}

#[test]
fn a_negative_modulus_bounds_the_remainder_below_zero() {
    let mut f = fixture();
    let (minus_four, minus_three, zero) = (f.c(-4), f.c(-3), f.c(0));
    let remainder = f.arena.floor_mod(f.x, minus_four);
    let at_least = f.arena.max(remainder, minus_three);
    let at_most = f.arena.min(remainder, zero);

    assert_eq!(f.simplify(at_least), remainder);
    assert_eq!(f.simplify(at_most), remainder);
}

#[test]
fn fusing_keeps_the_other_terms() {
    let mut f = fixture();
    let eight = f.c(8);
    let quotient = f.arena.floor_div(f.x, eight);
    let scaled = f.arena.mul(quotient, eight);
    let remainder = f.arena.floor_mod(f.x, eight);
    // The pair meets the other term in one sum: `(q * 8 + inner) + r`.
    let scaled_and_inner = f.arena.add(scaled, f.inner);
    let with_inner = f.arena.add(scaled_and_inner, remainder);

    let expected = f.arena.add(f.x, f.inner);
    assert_eq!(f.simplify(with_inner), expected);
}
