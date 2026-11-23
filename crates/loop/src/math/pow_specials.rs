//! C's special cases of `pow`, as libm's `pow` and Rust's `powf` give them,
//! laid over the general result by a chain of selects; the last applying
//! wins.

use super::builder::ProgramBuilder;
use super::program::{Comparison, FloatSlot, IntOp, IntSlot, PredSlot};

/// The f32 sign bit.
const SIGN_BIT: i32 = i32::MIN;

/// From here up, every f32 is an even integer.
const ALL_EVEN: f32 = 16_777_216.0;

/// `power`, which is `|x|^y` for a finite nonzero `x`, signed and with
/// every special case applied.
pub(super) fn with_specials(
    code: &mut ProgramBuilder,
    (x, y): (FloatSlot, FloatSlot),
    power: FloatSlot,
) -> FloatSlot {
    let (is_integer, is_odd) = parity(code, y);
    let magnitude = magnitude(code, x);
    let result = signed(code, (x, is_odd), power);
    let limit = limit(code, magnitude, y);
    let limit = signed(code, (x, is_odd), limit);
    let infinity = code.float(f32::INFINITY);
    let zero = code.float(0.0);
    let x_zero = code.compare(Comparison::Eq, magnitude, zero);
    let result = code.select(x_zero, limit, result);
    let x_infinite = code.compare(Comparison::Eq, magnitude, infinity);
    let result = code.select(x_infinite, limit, result);
    let y_magnitude = self::magnitude(code, y);
    let y_infinite = code.compare(Comparison::Eq, y_magnitude, infinity);
    let result = code.select(y_infinite, limit, result);
    let result = negative_base(code, (x, is_integer), result);
    let result = nans(code, (x, y), result);
    ones(code, (x, y), (magnitude, y_infinite), result)
}

/// Whether `y` is an integer, and whether it is an odd one.
fn parity(code: &mut ProgramBuilder, y: FloatSlot) -> (PredSlot, PredSlot) {
    let rounded = code.round_to_integer(y);
    let is_integer = code.compare(Comparison::Eq, rounded, y);
    let magnitude = magnitude(code, y);
    let all_even = code.float(ALL_EVEN);
    let small = code.compare(Comparison::Lt, magnitude, all_even);
    let n = code.float_to_int(rounded);
    let one = code.int(1);
    let low_bit = code.int_op(IntOp::And, n, one);
    let zero = code.int(0);
    let odd = code.select_int(small, low_bit, zero);
    let odd = code.select_int(is_integer, odd, zero);
    (is_integer, code.compare_int(Comparison::Eq, odd, one))
}

/// `|v|`.
fn magnitude(code: &mut ProgramBuilder, v: FloatSlot) -> FloatSlot {
    let bits = code.bits(v);
    let mask = code.int(!SIGN_BIT);
    let magnitude = code.int_op(IntOp::And, bits, mask);
    code.float_from_bits(magnitude)
}

/// `value`, a non-negative magnitude, negated where `x` is negative and
/// `y` an odd integer.
fn signed(
    code: &mut ProgramBuilder,
    (x, is_odd): (FloatSlot, PredSlot),
    value: FloatSlot,
) -> FloatSlot {
    let x_bits = code.bits(x);
    let sign_mask = code.int(SIGN_BIT);
    let sign = code.int_op(IntOp::And, x_bits, sign_mask);
    let zero = code.int(0);
    let flip: IntSlot = code.select_int(is_odd, sign, zero);
    let bits = code.bits(value);
    let signed = code.int_op(IntOp::Or, bits, flip);
    code.float_from_bits(signed)
}

/// `|x|^y` where `|x|` is 0 or infinite or `y` is infinite: infinite when
/// `|x| > 1` and `y > 0` agree, else zero.
fn limit(code: &mut ProgramBuilder, magnitude: FloatSlot, y: FloatSlot) -> FloatSlot {
    let (zero, one, infinity) = (code.float(0.0), code.float(1.0), code.float(f32::INFINITY));
    let above_one = code.compare(Comparison::Gt, magnitude, one);
    let positive = code.compare(Comparison::Gt, y, zero);
    let if_above = code.select(positive, infinity, zero);
    let if_below = code.select(positive, zero, infinity);
    code.select(above_one, if_above, if_below)
}

/// NaN for a finite negative `x` and a non-integral `y`.
fn negative_base(
    code: &mut ProgramBuilder,
    (x, is_integer): (FloatSlot, PredSlot),
    result: FloatSlot,
) -> FloatSlot {
    let nan = code.float(f32::NAN);
    let undefined = code.select(is_integer, result, nan);
    let minus_infinity = code.float(f32::NEG_INFINITY);
    let finite = code.compare(Comparison::Gt, x, minus_infinity);
    let finite_case = code.select(finite, undefined, result);
    let zero = code.float(0.0);
    let negative = code.compare(Comparison::Lt, x, zero);
    code.select(negative, finite_case, result)
}

/// NaN where either argument is.
fn nans(code: &mut ProgramBuilder, (x, y): (FloatSlot, FloatSlot), result: FloatSlot) -> FloatSlot {
    let nan = code.float(f32::NAN);
    let x_ordered = code.compare(Comparison::Eq, x, x);
    let result = code.select(x_ordered, result, nan);
    let y_ordered = code.compare(Comparison::Eq, y, y);
    code.select(y_ordered, result, nan)
}

/// 1 for `|x| = 1` with an infinite `y`, for `x = 1`, and for `y = ±0`,
/// NaNs included.
fn ones(
    code: &mut ProgramBuilder,
    (x, y): (FloatSlot, FloatSlot),
    (magnitude, y_infinite): (FloatSlot, PredSlot),
    result: FloatSlot,
) -> FloatSlot {
    let one = code.float(1.0);
    let unit = code.compare(Comparison::Eq, magnitude, one);
    let unit_case = code.select(y_infinite, one, result);
    let result = code.select(unit, unit_case, result);
    let x_one = code.compare(Comparison::Eq, x, one);
    let result = code.select(x_one, one, result);
    let zero = code.float(0.0);
    let y_zero = code.compare(Comparison::Eq, y, zero);
    code.select(y_zero, one, result)
}
