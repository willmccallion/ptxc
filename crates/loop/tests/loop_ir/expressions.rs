//! Index expressions: hash-consing and exact evaluation.

use ptxc_core::Idx;
use ptxc_loop::{Expr, ExprArena, VarId};

#[test]
fn equal_expressions_share_an_id() {
    let mut arena = ExprArena::new();
    let (x, one) = (arena.var(VarId::new(0)), arena.constant(1));

    let first = arena.add(x, one);
    let second = arena.add(x, one);
    let swapped = arena.add(one, x);

    assert_eq!(first, second);
    assert_ne!(first, swapped);
    assert_eq!(arena.get(first), Expr::Add(x, one));
}

#[test]
fn division_and_modulo_round_toward_negative_infinity() {
    let mut arena = ExprArena::new();
    let eval = |arena: &mut ExprArena, a: i64, b: i64| {
        let (a, b) = (arena.constant(a), arena.constant(b));
        let (quotient, remainder) = (arena.floor_div(a, b), arena.floor_mod(a, b));
        let none = |_| None;
        (arena.evaluate(quotient, &none), arena.evaluate(remainder, &none))
    };

    assert_eq!(eval(&mut arena, 7, 2), (Some(3), Some(1)));
    assert_eq!(eval(&mut arena, -7, 2), (Some(-4), Some(1)));
    assert_eq!(eval(&mut arena, 7, -2), (Some(-4), Some(-1)));
    assert_eq!(eval(&mut arena, -7, -2), (Some(3), Some(-1)));
    assert_eq!(eval(&mut arena, 7, 0), (None, None));
    assert_eq!(eval(&mut arena, i64::MIN, -1), (None, Some(0)));
    assert_eq!(eval(&mut arena, i64::MIN, 1), (Some(i64::MIN), Some(0)));
}

#[test]
fn evaluation_fails_on_overflow_and_unbound_variables() {
    let mut arena = ExprArena::new();
    let (x, big) = (arena.var(VarId::new(0)), arena.constant(i64::MAX));
    let overflow = arena.add(big, big);
    let none = |_| None;

    assert_eq!(arena.evaluate(overflow, &none), None);
    assert_eq!(arena.evaluate(x, &none), None);
    assert_eq!(arena.evaluate(x, &|_| Some(5)), Some(5));
}
