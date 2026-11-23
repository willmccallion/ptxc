//! Each step's result on the host, as the PTX codegen emits for it
//! computes.

#![allow(clippy::float_cmp, reason = "test code: exact values are the point")]

use super::builder::ProgramBuilder;
use super::program::{Comparison, FloatSlot, IntOp, IntSlot, PredSlot, Unary};

/// The program `body` builds from its argument, evaluated at `x`.
fn run(x: f32, body: impl FnOnce(&mut ProgramBuilder, FloatSlot) -> FloatSlot) -> f32 {
    let (mut code, argument) = ProgramBuilder::new();
    let result = body(&mut code, argument);
    code.finish::<Unary>(result).eval(x)
}

/// Whether the predicate `body` builds holds.
fn holds(body: impl FnOnce(&mut ProgramBuilder) -> PredSlot) -> bool {
    let result = run(0.0, |code, _| {
        let predicate = body(code);
        let (yes, no) = (code.float(1.0), code.float(0.0));
        code.select(predicate, yes, no)
    });
    result == 1.0
}

/// Whether the integer `body` builds equals `expected`.
fn int_is(expected: i32, body: impl FnOnce(&mut ProgramBuilder) -> IntSlot) -> bool {
    holds(|code| {
        let value = body(code);
        let expected = code.int(expected);
        code.compare_int(Comparison::Eq, value, expected)
    })
}

fn int_op(op: IntOp, a: i32, b: i32) -> impl FnOnce(&mut ProgramBuilder) -> IntSlot {
    move |code| {
        let (a, b) = (code.int(a), code.int(b));
        code.int_op(op, a, b)
    }
}

fn compare(cmp: Comparison, a: f32, b: f32) -> bool {
    holds(|code| {
        let (a, b) = (code.float(a), code.float(b));
        code.compare(cmp, a, b)
    })
}

#[test]
fn integer_operations_wrap_and_shift_by_kind() {
    assert!(int_is(0b1110, int_op(IntOp::Or, 0b1100, 0b1010)));
    assert!(int_is(0b1000, int_op(IntOp::And, 0b1100, 0b1010)));
    assert!(int_is(i32::MIN, int_op(IntOp::Add, i32::MAX, 1)));
    assert!(int_is(i32::MAX, int_op(IntOp::Sub, i32::MIN, 1)));
    assert!(int_is(i32::MIN, int_op(IntOp::Shl, 1, 31)));
    assert!(int_is(-4, int_op(IntOp::ShrArithmetic, -8, 1)));
    assert!(int_is(1, int_op(IntOp::ShrLogical, i32::MIN, 31)));
}

#[test]
fn comparisons_are_strict_ordered_and_signed() {
    assert!(!compare(Comparison::Lt, 1.0, 1.0));
    assert!(!compare(Comparison::Gt, 1.0, 1.0));
    assert!(compare(Comparison::Ge, 1.0, 1.0));
    assert!(compare(Comparison::Eq, 0.0, -0.0));
    assert!(!compare(Comparison::Ge, f32::NAN, f32::NAN));
    assert!(!compare(Comparison::Lt, f32::NAN, 1.0));
    assert!(holds(|code| {
        let (minus_one, zero) = (code.int(-1), code.int(0));
        code.compare_int(Comparison::Lt, minus_one, zero)
    }));
}

#[test]
fn conversions_round_to_nearest_even_and_saturate() {
    assert_eq!(run(2.5, ProgramBuilder::round_to_integer), 2.0);
    assert_eq!(run(-3.5, ProgramBuilder::round_to_integer), -4.0);
    assert!(int_is(-2, |code| {
        let x = code.float(-2.0);
        code.float_to_int(x)
    }));
    assert!(int_is(i32::MAX, |code| {
        let x = code.float(3e9);
        code.float_to_int(x)
    }));
    assert!(int_is(0, |code| {
        let x = code.float(f32::NAN);
        code.float_to_int(x)
    }));
    let rounded = run(0.0, |code, _| {
        let n = code.int((1 << 24) + 3);
        code.int_to_float(n)
    });
    assert_eq!(rounded, 16_777_220.0);
}

#[test]
fn bits_round_trip_and_fma_rounds_once() {
    assert!(int_is(0x3F80_0000, |code| {
        let one = code.float(1.0);
        code.bits(one)
    }));
    assert_eq!(
        run(0.0, |code, _| {
            let bits = code.int(0x4000_0000);
            code.float_from_bits(bits)
        }),
        2.0
    );
    let near_one = 1.0 + f32::EPSILON;
    let fused = run(near_one, |code, x| {
        let minus_one = code.float(-1.0);
        let x_minus = code.float(2.0 - near_one);
        code.fma(x, x_minus, minus_one)
    });
    assert_eq!(fused, -f32::EPSILON * f32::EPSILON);
}

#[test]
fn polynomials_evaluate_by_horner() {
    assert_eq!(run(2.0, |code, x| code.polynomial(x, 1.0, &[3.0, 5.0])), 27.0);
    assert_eq!(run(2.0, |code, x| code.polynomial(x, 7.0, &[])), 7.0);
}

/// The table's first two words after the zero word are 2/π to f64's
/// precision.
#[test]
fn the_reduction_table_holds_two_over_pi() {
    let [_, first, second, ..] = super::reduce::TWO_OVER_PI;
    let leading = f64::from(second).mul_add(2f64.powi(-64), f64::from(first) * 2f64.powi(-32));

    assert!((leading - std::f64::consts::FRAC_2_PI).abs() <= f64::EPSILON / 2.0);
}

#[test]
fn shifts_by_32_or_more_clamp_as_ptx_does() {
    assert!(int_is(0, int_op(IntOp::Shl, 1, 32)));
    assert!(int_is(0, int_op(IntOp::ShrLogical, -1, 32)));
    assert!(int_is(-1, int_op(IntOp::ShrArithmetic, i32::MIN, 32)));
    assert!(int_is(0, int_op(IntOp::ShrArithmetic, i32::MAX, 40)));
    assert!(int_is(-1, int_op(IntOp::ShrArithmetic, i32::MIN, 31)));
}

#[test]
fn products_xor_and_leading_zeros_are_unsigned_bit_operations() {
    assert!(int_is(-2, int_op(IntOp::Mul, i32::MAX, 2)));
    assert!(int_is(i32::MAX - 1, int_op(IntOp::MulHiUnsigned, -1, i32::MAX)));
    assert!(int_is(0b0110, int_op(IntOp::Xor, 0b1100, 0b1010)));
    assert!(int_is(31, |code| {
        let one = code.int(1);
        code.leading_zeros(one)
    }));
    assert!(int_is(32, |code| {
        let zero = code.int(0);
        code.leading_zeros(zero)
    }));
    assert!(holds(|code| {
        let (minus_one, zero) = (code.int(-1), code.int(0));
        code.compare_unsigned(Comparison::Gt, minus_one, zero)
    }));
    let rounded = run(0.0, |code, _| {
        let n = code.int(-1);
        code.unsigned_to_float(n)
    });
    assert_eq!(rounded, 4_294_967_296.0);
}
