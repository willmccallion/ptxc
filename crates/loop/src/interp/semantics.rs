//! The scalar semantics the interpreter gives each operation, case by case.

#![allow(clippy::unwrap_used, clippy::float_cmp, reason = "test code: exact values are the point")]

use super::RunError;
use super::ops::{binary, cast, unary};
use super::value::{Value, round, saturate, wrap};
use crate::compute::{BinaryOp, InfSign, UnaryOp};
use ptxc_core::DType;
use ptxc_core::float::Overflow;

const fn f(value: f64) -> Value {
    Value::Float(value)
}

const fn i(value: i64) -> Value {
    Value::Int(value)
}

const fn b(value: bool) -> Value {
    Value::Bool(value)
}

fn is_nan(value: Value) -> bool {
    matches!(value, Value::Float(x) if x.is_nan())
}

#[test]
fn integer_unary_ops_wrap_to_their_width() {
    assert_eq!(unary(UnaryOp::Abs, i(-128), DType::I8), i(-128));
    assert_eq!(unary(UnaryOp::Abs, i(-5), DType::I32), i(5));
    assert_eq!(unary(UnaryOp::Neg, i(5), DType::I32), i(-5));
    assert_eq!(unary(UnaryOp::Neg, i(i64::from(i32::MIN)), DType::I32), i(i64::from(i32::MIN)));
    assert_eq!(unary(UnaryOp::Sign, i(-7), DType::I64), i(-1));
    assert_eq!(unary(UnaryOp::Sign, i(0), DType::I64), i(0));
    assert_eq!(unary(UnaryOp::Not, b(true), DType::Bool), b(false));
    assert_eq!(unary(UnaryOp::Not, b(false), DType::Bool), b(true));
}

#[test]
fn float_unary_ops_round_to_their_type() {
    let cases = [
        (UnaryOp::Abs, -2.5, 2.5),
        (UnaryOp::Neg, 1.5, -1.5),
        (UnaryOp::Exp, 0.0, 1.0),
        (UnaryOp::Expm1, 0.0, 0.0),
        (UnaryOp::Log1p, 0.0, 0.0),
        (UnaryOp::Log, 1.0, 0.0),
        (UnaryOp::Sqrt, 4.0, 2.0),
        (UnaryOp::Reciprocal, 4.0, 0.25),
        (UnaryOp::Floor, -1.5, -2.0),
        (UnaryOp::Ceil, -1.5, -1.0),
        (UnaryOp::Round, 2.5, 2.0),
        (UnaryOp::Round, 3.5, 4.0),
        (UnaryOp::Sign, -3.0, -1.0),
        (UnaryOp::Sign, 3.0, 1.0),
        (UnaryOp::Erf, 0.0, 0.0),
        (UnaryOp::Sin, 0.0, 0.0),
        (UnaryOp::Cos, 0.0, 1.0),
        (UnaryOp::Tanh, 0.0, 0.0),
    ];
    for (op, input, expected) in cases {
        assert_eq!(unary(op, f(input), DType::F32), f(expected), "{op:?}({input})");
    }
    assert_eq!(unary(UnaryOp::Exp, f(1.0), DType::F16), f(2.718_75));
    let tiny = 1e-10;
    assert_eq!(
        unary(UnaryOp::Expm1, f(tiny), DType::F32),
        round(tiny.exp_m1(), DType::F32, Overflow::NonFinite)
    );
    assert_eq!(
        unary(UnaryOp::Log1p, f(tiny), DType::F32),
        round(tiny.ln_1p(), DType::F32, Overflow::NonFinite)
    );
    assert_ne!(unary(UnaryOp::Expm1, f(tiny), DType::F32), f(0.0));
    assert!(is_nan(unary(UnaryOp::Sign, f(f64::NAN), DType::F32)));
    let negative_zero = unary(UnaryOp::Sign, f(-0.0), DType::F32);
    assert!(matches!(negative_zero, Value::Float(x) if x == 0.0 && x.is_sign_negative()));
}

#[test]
fn float_tests_give_booleans() {
    let test = |op, value| unary(op, f(value), DType::F32);
    assert_eq!(test(UnaryOp::IsNaN, f64::NAN), b(true));
    assert_eq!(test(UnaryOp::IsNaN, 1.0), b(false));
    assert_eq!(test(UnaryOp::IsInf(InfSign::Either), f64::NEG_INFINITY), b(true));
    assert_eq!(test(UnaryOp::IsInf(InfSign::Positive), f64::INFINITY), b(true));
    assert_eq!(test(UnaryOp::IsInf(InfSign::Positive), f64::NEG_INFINITY), b(false));
    assert_eq!(test(UnaryOp::IsInf(InfSign::Negative), f64::NEG_INFINITY), b(true));
    assert_eq!(test(UnaryOp::IsInf(InfSign::Negative), f64::INFINITY), b(false));
    assert_eq!(test(UnaryOp::IsInf(InfSign::Either), 1.0), b(false));
}

#[test]
fn float_binary_ops_follow_ieee_and_numpy() {
    let op = |op, x, y| binary(op, f(x), f(y), DType::F32).unwrap();
    assert_eq!(op(BinaryOp::Add, 1.5, 2.25), f(3.75));
    assert_eq!(op(BinaryOp::Sub, 1.0, 3.0), f(-2.0));
    assert_eq!(op(BinaryOp::Mul, 1.5, -4.0), f(-6.0));
    assert_eq!(op(BinaryOp::Div, 1.0, 4.0), f(0.25));
    assert_eq!(op(BinaryOp::Pow, 2.0, 10.0), f(1024.0));
    assert_eq!(op(BinaryOp::Max, 1.0, 2.0), f(2.0));
    assert_eq!(op(BinaryOp::Min, 1.0, 2.0), f(1.0));
    assert_eq!(op(BinaryOp::Min, 2.0, 1.0), f(1.0));
    assert!(is_nan(op(BinaryOp::Max, 1.0, f64::NAN)));
    assert!(is_nan(op(BinaryOp::Max, f64::NAN, 1.0)));
    assert!(is_nan(op(BinaryOp::Min, f64::NAN, 1.0)));
    assert!(is_nan(op(BinaryOp::Min, 1.0, f64::NAN)));
    assert_eq!(op(BinaryOp::TruncatedRem, -7.5, 2.0), f(-1.5));
    assert_eq!(
        binary(BinaryOp::Add, f(1.0), f(f64::from(f32::EPSILON) / 4.0), DType::F32).unwrap(),
        f(1.0)
    );
}

#[test]
fn integer_binary_ops_wrap_and_truncate() {
    let op = |op, x, y, dtype| binary(op, i(x), i(y), dtype).unwrap();
    assert_eq!(op(BinaryOp::Add, 127, 1, DType::I8), i(-128));
    assert_eq!(op(BinaryOp::Sub, 0, 1, DType::U8), i(255));
    assert_eq!(op(BinaryOp::Mul, 1 << 20, 1 << 12, DType::I32), i(0));
    assert_eq!(op(BinaryOp::Mul, 6, 7, DType::I32), i(42));
    assert_eq!(op(BinaryOp::Div, -7, 2, DType::I32), i(-3));
    assert_eq!(op(BinaryOp::Pow, 3, 5, DType::I32), i(243));
    assert_eq!(op(BinaryOp::Pow, 2, 31, DType::I32), i(i64::from(i32::MIN)));
    assert_eq!(op(BinaryOp::Pow, 2, -1, DType::I32), i(0));
    assert_eq!(op(BinaryOp::Max, -7, 2, DType::I32), i(2));
    assert_eq!(op(BinaryOp::Min, -7, 2, DType::I32), i(-7));
    assert_eq!(op(BinaryOp::FlooredRem, -7, 2, DType::I32), i(1));
    assert_eq!(op(BinaryOp::FlooredRem, 7, -2, DType::I32), i(-1));
    assert_eq!(op(BinaryOp::FlooredRem, 7, 2, DType::I32), i(1));
    assert_eq!(op(BinaryOp::FlooredRem, 6, -3, DType::I32), i(0));
    assert_eq!(op(BinaryOp::TruncatedRem, -7, 2, DType::I32), i(-1));
    for divide in [BinaryOp::Div, BinaryOp::FlooredRem, BinaryOp::TruncatedRem] {
        assert_eq!(binary(divide, i(1), i(0), DType::I32), Err(RunError::DivisionByZero));
    }
}

#[test]
fn comparisons_give_booleans_and_nan_compares_false() {
    let cmp = |op, x, y, dtype| binary(op, x, y, dtype).unwrap();
    assert_eq!(cmp(BinaryOp::Equal, f(1.0), f(1.0), DType::F32), b(true));
    assert_eq!(cmp(BinaryOp::Equal, f(f64::NAN), f(f64::NAN), DType::F32), b(false));
    assert_eq!(cmp(BinaryOp::LessOrEqual, i(2), i(2), DType::I32), b(true));
    assert_eq!(cmp(BinaryOp::LessOrEqual, i(3), i(2), DType::I32), b(false));
    assert_eq!(cmp(BinaryOp::GreaterOrEqual, i(2), i(3), DType::I32), b(false));
    assert_eq!(cmp(BinaryOp::GreaterOrEqual, f(f64::NAN), f(1.0), DType::F32), b(false));
    assert_eq!(cmp(BinaryOp::Less, f(1.0), f(2.0), DType::F32), b(true));
    assert_eq!(cmp(BinaryOp::Greater, b(true), b(false), DType::Bool), b(true));
    assert_eq!(cmp(BinaryOp::Equal, b(true), b(false), DType::Bool), b(false));
}

#[test]
fn boolean_ops_combine_truth_values() {
    let op = |op, x, y| binary(op, b(x), b(y), DType::Bool).unwrap();
    assert_eq!(op(BinaryOp::And, true, false), b(false));
    assert_eq!(op(BinaryOp::Or, true, false), b(true));
    assert_eq!(op(BinaryOp::Or, true, true), b(true));
    assert_eq!(op(BinaryOp::Max, true, true), b(true));
    assert_eq!(op(BinaryOp::Xor, true, true), b(false));
    assert_eq!(op(BinaryOp::Xor, true, false), b(true));
    assert_eq!(op(BinaryOp::Max, false, true), b(true));
    assert_eq!(op(BinaryOp::Min, false, true), b(false));
}

#[test]
fn casts_round_wrap_or_saturate_by_direction() {
    assert_eq!(cast(f(300.7), DType::F32, DType::I8, Overflow::Saturate), i(127));
    assert_eq!(cast(f(-300.0), DType::F32, DType::I8, Overflow::Saturate), i(-128));
    assert_eq!(cast(f(-5.5), DType::F32, DType::U8, Overflow::Saturate), i(0));
    assert_eq!(cast(f(f64::NAN), DType::F32, DType::I32, Overflow::Saturate), i(0));
    assert_eq!(cast(f(-2.9), DType::F32, DType::I32, Overflow::Saturate), i(-2));
    assert_eq!(cast(i(300), DType::I32, DType::I8, Overflow::Saturate), i(44));
    assert_eq!(cast(i(16_777_217), DType::I32, DType::F32, Overflow::Saturate), f(16_777_216.0));
    assert_eq!(cast(f(0.0), DType::F32, DType::Bool, Overflow::Saturate), b(false));
    assert_eq!(cast(f(-0.5), DType::F32, DType::Bool, Overflow::Saturate), b(true));
    assert_eq!(cast(i(0), DType::I32, DType::Bool, Overflow::Saturate), b(false));
    assert_eq!(cast(b(true), DType::Bool, DType::F32, Overflow::Saturate), f(1.0));
    assert_eq!(cast(f(1000.0), DType::F32, DType::F8E4M3Fn, Overflow::Saturate), f(448.0));
    assert!(is_nan(cast(f(1000.0), DType::F32, DType::F8E4M3Fn, Overflow::NonFinite)));
}

#[test]
fn values_wrap_saturate_and_round_to_their_type() {
    assert_eq!(wrap(1 << 31, DType::I32), i(i64::from(i32::MIN)));
    assert_eq!(wrap(256, DType::U8), i(0));
    assert_eq!(wrap(-1, DType::U8), i(255));
    assert_eq!(wrap(-129, DType::I8), i(127));
    assert_eq!(wrap(128, DType::I8), i(-128));
    assert_eq!(wrap(127, DType::I8), i(127));
    assert_eq!(wrap(200, DType::U8), i(200));
    assert_eq!(wrap(i128::from(i64::MAX) + 1, DType::I64), i(i64::MIN));
    assert_eq!(saturate(1 << 40, DType::I32), i(i64::from(i32::MAX)));
    assert_eq!(saturate(-200, DType::I8), i(-128));
    assert_eq!(saturate(300, DType::U8), i(255));
    assert_eq!(saturate(-1, DType::U8), i(0));
    assert_eq!(round(0.1, DType::Bf16, Overflow::Saturate), f(0.100_097_656_25));
    assert_eq!(round(2.0, DType::Bool, Overflow::Saturate), b(true));
    assert_eq!(round(0.0, DType::Bool, Overflow::Saturate), b(false));
    assert_eq!(round(-2.7, DType::I32, Overflow::Saturate), i(-2));
    assert!(!f(0.0).bool());
    assert!(f(0.5).bool());
    assert!(!i(0).bool());
    assert!(i(-3).bool());
    assert_eq!(b(true).int(), 1);
    assert_eq!(b(false).float(), 0.0);
    assert_eq!(i(7).float(), 7.0);
    assert_eq!(f(-7.9).int(), -7);
}
