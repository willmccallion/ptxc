//! Unary, binary and variadic ops, `Where`, and casts.

use crate::builders::{assert_close, assert_exact, bools, f32s, i64s, op, tensor, try_run, values};
use ptxc_core::float::F16;
use ptxc_interp::InterpError;

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

#[test]
fn float_unary_ops_follow_their_definitions() {
    let x = || f32s(&[4], &[-1.5, 0.25, 2.5, 4.0]);
    let unary = |name: &str| op(name, "f32[4]", &[x()]);

    assert_close(&unary("abs"), &[4], &[1.5, 0.25, 2.5, 4.0]);
    assert_close(&unary("neg"), &[4], &[1.5, -0.25, -2.5, -4.0]);
    assert_close(
        &unary("exp"),
        &[4],
        &[(-1.5f64).exp(), 0.25f64.exp(), 2.5f64.exp(), 4.0f64.exp()],
    );
    assert_close(&unary("floor"), &[4], &[-2.0, 0.0, 2.0, 4.0]);
    assert_close(&unary("ceil"), &[4], &[-1.0, 1.0, 3.0, 4.0]);
    assert_close(&unary("round"), &[4], &[-2.0, 0.0, 2.0, 4.0]);
    assert_close(&unary("sign"), &[4], &[-1.0, 1.0, 1.0, 1.0]);
    assert_close(&unary("reciprocal"), &[4], &[-1.0 / 1.5, 4.0, 0.4, 0.25]);
    assert_close(
        &unary("sin"),
        &[4],
        &[(-1.5f64).sin(), 0.25f64.sin(), 2.5f64.sin(), 4.0f64.sin()],
    );
    assert_close(
        &unary("cos"),
        &[4],
        &[(-1.5f64).cos(), 0.25f64.cos(), 2.5f64.cos(), 4.0f64.cos()],
    );
}

#[test]
fn logarithms_roots_and_erf_follow_their_definitions() {
    let x = || f32s(&[3], &[0.5, 1.0, 4.0]);

    assert_close(&op("log", "f32[3]", &[x()]), &[3], &[0.5f64.ln(), 0.0, 4.0f64.ln()]);
    assert_close(&op("sqrt", "f32[3]", &[x()]), &[3], &[0.5f64.sqrt(), 1.0, 2.0]);
    assert_close(
        &op("erf", "f32[3]", &[x()]),
        &[3],
        &[0.520_499_877_813_046_5, 0.842_700_792_949_714_9, 0.999_999_984_582_742_1],
    );
}

#[test]
fn sign_of_zero_and_nan_and_rounding_ties_to_even() {
    let x = f32s(&[4], &[0.0, f32::NAN, 0.5, -2.5]);

    assert_close(&op("sign", "f32[4]", std::slice::from_ref(&x)), &[4], &[0.0, NAN, 1.0, -1.0]);
    assert_close(&op("round", "f32[4]", &[x]), &[4], &[0.0, NAN, 0.0, -2.0]);
}

#[test]
fn nan_and_infinity_tests_give_bools() {
    let x = || f32s(&[4], &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0]);

    assert_exact(&op("is_nan", "bool[4]", &[x()]), &[4], &[1, 0, 0, 0]);
    assert_exact(&op("is_inf<detect=both>", "bool[4]", &[x()]), &[4], &[0, 1, 1, 0]);
    assert_exact(&op("is_inf<detect=positive>", "bool[4]", &[x()]), &[4], &[0, 1, 0, 0]);
    assert_exact(&op("is_inf<detect=negative>", "bool[4]", &[x()]), &[4], &[0, 0, 1, 0]);
    assert_exact(&op("is_inf<detect=neither>", "bool[4]", &[x()]), &[4], &[0, 0, 0, 0]);
}

#[test]
fn integer_unary_ops_wrap_and_not_negates_bools() {
    let small = tensor(&[3], vec![-128_i8, -3, 5]);

    assert_exact(&op("abs", "i8[3]", std::slice::from_ref(&small)), &[3], &[-128, 3, 5]);
    assert_exact(&op("neg", "i8[3]", std::slice::from_ref(&small)), &[3], &[-128, 3, -5]);
    assert_exact(&op("sign", "i8[3]", &[small]), &[3], &[-1, -1, 1]);
    assert_exact(&op("not", "bool[2]", &[bools(&[2], &[true, false])]), &[2], &[0, 1]);
}

#[test]
fn float_binary_ops_broadcast_both_ways() {
    let a = || f32s(&[2, 1], &[6.0, -7.5]);
    let b = || f32s(&[3], &[2.0, -4.0, 0.5]);
    let binary = |name: &str| op(name, "f32[2, 3]", &[a(), b()]);

    assert_close(&binary("add"), &[2, 3], &[8.0, 2.0, 6.5, -5.5, -11.5, -7.0]);
    assert_close(&binary("sub"), &[2, 3], &[4.0, 10.0, 5.5, -9.5, -3.5, -8.0]);
    assert_close(&binary("mul"), &[2, 3], &[12.0, -24.0, 3.0, -15.0, 30.0, -3.75]);
    assert_close(&binary("div"), &[2, 3], &[3.0, -1.5, 12.0, -3.75, 1.875, -15.0]);
    assert_close(&binary("mod<kind=fmod>"), &[2, 3], &[0.0, 2.0, 0.0, -1.5, -3.5, -0.0]);
    assert_close(
        &op("pow", "f32[2, 3]", &[a(), f32s(&[3], &[2.0, 0.5, -1.0])]),
        &[2, 3],
        &[36.0, 6f64.sqrt(), 1.0 / 6.0, 56.25, NAN, -1.0 / 7.5],
    );
}

#[test]
fn comparisons_give_bools_for_floats_and_integers() {
    let floats = || [f32s(&[3], &[1.0, 2.0, f32::NAN]), f32s(&[], &[2.0])];
    let ints = || [i64s(&[3], &[1, 2, 3]), i64s(&[], &[2])];

    assert_exact(&op("equal", "bool[3]", &floats()), &[3], &[0, 1, 0]);
    assert_exact(&op("less", "bool[3]", &floats()), &[3], &[1, 0, 0]);
    assert_exact(&op("less_or_equal", "bool[3]", &ints()), &[3], &[1, 1, 0]);
    assert_exact(&op("greater", "bool[3]", &ints()), &[3], &[0, 0, 1]);
    assert_exact(&op("greater_or_equal", "bool[3]", &ints()), &[3], &[0, 1, 1]);
    assert_exact(&op("equal", "bool[3]", &ints()), &[3], &[0, 1, 0]);
}

#[test]
fn logic_ops_follow_their_truth_tables() {
    let operands =
        || [bools(&[4], &[false, false, true, true]), bools(&[4], &[false, true, false, true])];

    assert_exact(&op("and", "bool[4]", &operands()), &[4], &[0, 0, 0, 1]);
    assert_exact(&op("or", "bool[4]", &operands()), &[4], &[0, 1, 1, 1]);
    assert_exact(&op("xor", "bool[4]", &operands()), &[4], &[0, 1, 1, 0]);
    assert_exact(&op("equal", "bool[4]", &operands()), &[4], &[1, 0, 0, 1]);
}

#[test]
fn integer_arithmetic_wraps_and_divides_toward_zero() {
    let a = || tensor(&[4], vec![100_i8, -7, 7, -128]);
    let b = || tensor(&[4], vec![100_i8, 2, -2, -1]);

    assert_exact(&op("add", "i8[4]", &[a(), b()]), &[4], &[-56, -5, 5, 127]);
    assert_exact(&op("sub", "i8[4]", &[a(), b()]), &[4], &[0, -9, 9, -127]);
    assert_exact(&op("mul", "i8[4]", &[a(), b()]), &[4], &[16, -14, -14, -128]);
    assert_exact(&op("div", "i8[4]", &[a(), b()]), &[4], &[1, -3, -3, -128]);
}

#[test]
fn integer_remainders_take_the_divisor_or_the_dividend_sign() {
    let a = || i64s(&[4], &[7, -7, 7, -7]);
    let b = || i64s(&[4], &[3, 3, -3, -3]);

    assert_exact(&op("mod<kind=integer>", "i64[4]", &[a(), b()]), &[4], &[1, 2, -2, -1]);
    assert_exact(&op("mod<kind=fmod>", "i64[4]", &[a(), b()]), &[4], &[1, -1, 1, -1]);
    assert_exact(
        &op("mod<kind=integer>", "i64[2]", &[i64s(&[2], &[6, i64::MIN]), i64s(&[2], &[-3, -1])]),
        &[2],
        &[0, 0],
    );
}

#[test]
fn integer_division_by_zero_is_an_error() {
    for name in ["div", "mod<kind=integer>", "mod<kind=fmod>"] {
        let text = format!(
            "graph(%a: i64[1], %b: i64[1]) {{\n  %y = {name} %a, %b : i64[1]\n  return %y\n}}\n"
        );

        let error = try_run(&text, &[i64s(&[1], &[1]), i64s(&[1], &[0])]).unwrap_err();

        assert!(matches!(error, InterpError::IntegerDivisionByZero { .. }), "{name}: {error}");
    }
}

#[test]
fn powers_mix_integer_and_float_operands() {
    let int_int = op("pow", "i64[4]", &[i64s(&[4], &[3, 2, 2, -2]), i64s(&[4], &[4, 0, -1, 3])]);
    let int_float = op("pow", "i64[2]", &[i64s(&[2], &[2, 9]), f32s(&[2], &[0.5, 0.5])]);
    let float_int = op("pow", "f32[2]", &[f32s(&[2], &[1.5, 4.0]), i64s(&[2], &[2, -1])]);

    assert_exact(&int_int, &[4], &[81, 1, 0, -8]);
    assert_exact(&int_float, &[2], &[1, 3]);
    assert_close(&float_int, &[2], &[2.25, 0.25]);
}

#[test]
fn max_and_min_propagate_nan_and_mean_divides_the_sum() {
    let operands =
        || [f32s(&[3], &[1.0, 5.0, f32::NAN]), f32s(&[3], &[4.0, 2.0, 0.0]), f32s(&[], &[3.0])];

    assert_close(&op("max", "f32[3]", &operands()), &[3], &[4.0, 5.0, NAN]);
    assert_close(&op("min", "f32[3]", &operands()), &[3], &[1.0, 2.0, NAN]);
    assert_close(&op("sum", "f32[3]", &operands()), &[3], &[8.0, 10.0, NAN]);
    assert_close(
        &op("mean", "f32[2]", &[f32s(&[2], &[1.0, 2.0]), f32s(&[2], &[2.0, 7.0])]),
        &[2],
        &[1.5, 4.5],
    );
    let ints = [i64s(&[2], &[1, 9]), i64s(&[2], &[4, -2])];
    assert_exact(&op("max", "i64[2]", &ints), &[2], &[4, 9]);
    assert_exact(&op("min", "i64[2]", &ints), &[2], &[1, -2]);
}

#[test]
fn where_selects_by_a_broadcast_condition_for_any_type() {
    let condition = bools(&[2, 1], &[true, false]);

    let floats =
        op("where", "f32[2, 2]", &[condition.clone(), f32s(&[2], &[1.0, 2.0]), f32s(&[], &[9.0])]);
    let ints = op(
        "where",
        "i8[2, 2]",
        &[condition, tensor(&[], vec![5_i8]), tensor(&[2, 2], vec![1_i8, 2, 3, 4])],
    );

    assert_close(&floats, &[2, 2], &[1.0, 2.0, 9.0, 9.0]);
    assert_exact(&ints, &[2, 2], &[5, 5, 3, 4]);
}

#[test]
fn float_to_integer_casts_truncate_and_saturate() {
    let x = f32s(&[5], &[2.9, -2.9, 300.0, -300.0, f32::NAN]);

    assert_exact(
        &op("cast<to=i8>", "i8[5]", std::slice::from_ref(&x)),
        &[5],
        &[2, -2, 127, -128, 0],
    );
    assert_exact(&op("cast<to=u8>", "u8[5]", std::slice::from_ref(&x)), &[5], &[2, 0, 255, 0, 0]);
    assert_exact(&op("cast<to=bool>", "bool[5]", &[x]), &[5], &[1, 1, 1, 1, 1]);
}

#[test]
fn integer_casts_wrap_and_bools_become_zero_and_one() {
    let x = i64s(&[4], &[300, -1, 0, 70_000]);

    assert_exact(&op("cast<to=u8>", "u8[4]", std::slice::from_ref(&x)), &[4], &[44, 255, 0, 112]);
    assert_exact(
        &op("cast<to=i32>", "i32[4]", std::slice::from_ref(&x)),
        &[4],
        &[300, -1, 0, 70_000],
    );
    assert_exact(&op("cast<to=bool>", "bool[4]", std::slice::from_ref(&x)), &[4], &[1, 1, 0, 1]);
    assert_close(&op("cast<to=f16>", "f16[4]", &[x]), &[4], &[300.0, -1.0, 0.0, INF]);
    assert_close(&op("cast<to=f32>", "f32[2]", &[bools(&[2], &[true, false])]), &[2], &[1.0, 0.0]);
    assert_exact(&op("cast<to=i64>", "i64[2]", &[bools(&[2], &[true, false])]), &[2], &[1, 0]);
}

/// 2^62 + 2^38 + 1 and 2^62 + 2^54 + 1 lie just above midpoints of f32
/// and bf16, but their nearest f64s are the midpoints, which tie to even.
#[test]
fn integer_casts_to_floats_round_once() {
    let x = || i64s(&[2], &[(1 << 62) + (1 << 38) + 1, -((1 << 62) + (1 << 54) + 1)]);

    let as_f32 = op("cast<to=f32>", "f32[2]", &[x()]);
    let as_bf16 = op("cast<to=bf16>", "bf16[2]", &[x()]);

    assert_eq!(values(&as_f32)[0], 2f64.powi(62) + 2f64.powi(39));
    assert_eq!(values(&as_bf16)[1], -(2f64.powi(62) + 2f64.powi(55)));
}

#[test]
fn float_casts_round_to_nearest_even_and_fp8_honours_saturate() {
    let x = || f32s(&[3], &[1.000_488_3, 500.0, -1e6]);

    let half = op("cast<to=f16>", "f16[3]", &[x()]);
    let saturated = op("cast<to=f8e4m3fn, overflow=saturate>", "f8e4m3fn[3]", &[x()]);
    let non_finite = op("cast<to=f8e4m3fn, overflow=non_finite>", "f8e4m3fn[3]", &[x()]);
    let widened = op(
        "cast<to=f32>",
        "f32[2]",
        &[tensor(&[2], vec![F16::from_f32(0.5), F16::from_f32(-2.0)])],
    );

    assert_close(&half, &[3], &[1.0, 500.0, -INF]);
    assert_close(&saturated, &[3], &[1.0, 448.0, -448.0]);
    assert_eq!(values(&non_finite)[1..].iter().filter(|v| v.is_nan()).count(), 2);
    assert_close(&widened, &[2], &[0.5, -2.0]);
}

#[test]
fn e5m2_casts_saturate_or_overflow_to_infinity() {
    let x = || f32s(&[2], &[1.0, 1e6]);

    assert_close(
        &op("cast<to=f8e5m2, overflow=saturate>", "f8e5m2[2]", &[x()]),
        &[2],
        &[1.0, 57_344.0],
    );
    assert_close(
        &op("cast<to=f8e5m2, overflow=non_finite>", "f8e5m2[2]", &[x()]),
        &[2],
        &[1.0, INF],
    );
}

#[test]
fn integers_cast_to_fp8_round_once_and_saturate() {
    let x = || i64s(&[2], &[3, 100_000]);

    assert_close(
        &op("cast<to=f8e4m3fn, overflow=saturate>", "f8e4m3fn[2]", &[x()]),
        &[2],
        &[3.0, 448.0],
    );
    assert_close(
        &op("cast<to=f8e5m2, overflow=saturate>", "f8e5m2[2]", &[x()]),
        &[2],
        &[3.0, 57_344.0],
    );
}
