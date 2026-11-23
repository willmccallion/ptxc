//! `fmod`, the remainder of truncated division, which is exact. Each step
//! takes `r - q b` for `b = |y| 2^k`, with `k` leaving a quotient `q` of at
//! most 24 bits, so one fused step computes it exactly and the exponent gap
//! closes by 23 bits; enough steps close any gap between two f32s.

use super::builder::ProgramBuilder;
use super::constants::{BIAS, MANTISSA_BITS};
use super::exp::power_of_two;
use super::program::{Binary, Comparison, FloatSlot, IntOp, IntSlot, MathProgram};

/// Steps enough for the largest exponent gap, 276 between the largest
/// f32's and the least subnormal's. Each step leaves `r < b`, closing the
/// gap by at least 23 bits: eleven close it to 23, and a twelfth, with
/// `b = |y|`, leaves `r < |y|`. Random pairs need ten.
const STEPS: usize = 12;

/// The f32 sign bit.
const SIGN_BIT: i32 = i32::MIN;

/// The exponent a subnormal's leading zeros are counted from: a subnormal
/// with `z` leading zeros has exponent `SUBNORMAL_BASE - z`.
const SUBNORMAL_BASE: i32 = -118;

/// The largest power of two one scaling multiplies by.
const LARGEST_SCALE: i32 = 126;

pub(super) fn fmod() -> MathProgram<Binary> {
    let (mut code, x, y) = ProgramBuilder::new_binary();
    let x_bits = code.bits(x);
    let mut r = magnitude(&mut code, x_bits);
    let y_bits = code.bits(y);
    let divisor = magnitude(&mut code, y_bits);
    for _ in 0..STEPS {
        r = step(&mut code, r, divisor);
    }
    let r_bits = code.bits(r);
    let sign_mask = code.int(SIGN_BIT);
    let sign = code.int_op(IntOp::And, x_bits, sign_mask);
    let signed_bits = code.int_op(IntOp::Or, r_bits, sign);
    let signed = code.float_from_bits(signed_bits);
    let result = specials(&mut code, (x, y), signed);
    code.finish(result)
}

/// `|v|`, from `v`'s bits.
fn magnitude(code: &mut ProgramBuilder, bits: IntSlot) -> FloatSlot {
    let mask = code.int(!SIGN_BIT);
    let magnitude = code.int_op(IntOp::And, bits, mask);
    code.float_from_bits(magnitude)
}

/// `r` reduced modulo `b = divisor 2^k`, `k` as large as leaves the
/// quotient within 24 bits.
fn step(code: &mut ProgramBuilder, r: FloatSlot, divisor: FloatSlot) -> FloatSlot {
    let k = divisor_scale(code, r, divisor);
    let b = scale(code, divisor, k);
    let quotient = code.div(r, b);
    let q = code.round_to_integer(quotient);
    let zero = code.float(0.0);
    let minus_q = code.sub(zero, q);
    let reduced = code.fma(minus_q, b, r);
    let overshot = code.compare(Comparison::Lt, reduced, zero);
    let restored = code.add(reduced, b);
    code.select(overshot, restored, reduced)
}

/// How far to scale `divisor` up: the exponent gap to `r` less 23, so the
/// quotient keeps within 24 bits, or 0.
fn divisor_scale(code: &mut ProgramBuilder, r: FloatSlot, divisor: FloatSlot) -> IntSlot {
    let (r_exponent, divisor_exponent) = (exponent(code, r), exponent(code, divisor));
    let gap = code.int_op(IntOp::Sub, r_exponent, divisor_exponent);
    let precision = code.int(MANTISSA_BITS);
    let beyond = code.int_op(IntOp::Sub, gap, precision);
    let zero = code.int(0);
    let negative = code.compare_int(Comparison::Lt, beyond, zero);
    code.select_int(negative, zero, beyond)
}

/// The exponent of a non-negative `v`, subnormals included.
fn exponent(code: &mut ProgramBuilder, v: FloatSlot) -> IntSlot {
    let bits = code.bits(v);
    let shift = code.int(MANTISSA_BITS);
    let field = code.int_op(IntOp::ShrLogical, bits, shift);
    let bias = code.int(BIAS);
    let normal = code.int_op(IntOp::Sub, field, bias);
    let zeros = code.leading_zeros(bits);
    let base = code.int(SUBNORMAL_BASE);
    let subnormal = code.int_op(IntOp::Sub, base, zeros);
    let zero = code.int(0);
    let is_subnormal = code.compare_int(Comparison::Eq, field, zero);
    code.select_int(is_subnormal, subnormal, normal)
}

/// `v 2^k` for `0 <= k <= 276`, as three exact multiplications by normal
/// powers of two; `v 2^k` is at most the remainder, so none overflows.
fn scale(code: &mut ProgramBuilder, v: FloatSlot, k: IntSlot) -> FloatSlot {
    let mut scaled = v;
    let mut remaining = k;
    let largest = code.int(LARGEST_SCALE);
    for _ in 0..3 {
        let above = code.compare_int(Comparison::Gt, remaining, largest);
        let part = code.select_int(above, largest, remaining);
        let factor = power_of_two(code, part);
        scaled = code.mul(scaled, factor);
        remaining = code.int_op(IntOp::Sub, remaining, part);
    }
    scaled
}

/// `result` with C's special cases: `fmod(x, ±inf) = x` for finite `x`,
/// and NaN for `y = 0`, an infinite `x`, or a NaN.
fn specials(
    code: &mut ProgramBuilder,
    (x, y): (FloatSlot, FloatSlot),
    result: FloatSlot,
) -> FloatSlot {
    let infinity = code.float(f32::INFINITY);
    let x_bits = code.bits(x);
    let x_magnitude = magnitude(code, x_bits);
    let y_bits = code.bits(y);
    let y_magnitude = magnitude(code, y_bits);
    let y_infinite = code.compare(Comparison::Eq, y_magnitude, infinity);
    let result = code.select(y_infinite, x, result);
    let nan = code.float(f32::NAN);
    let zero = code.float(0.0);
    let y_zero = code.compare(Comparison::Eq, y_magnitude, zero);
    let result = code.select(y_zero, nan, result);
    let x_infinite = code.compare(Comparison::Eq, x_magnitude, infinity);
    let result = code.select(x_infinite, nan, result);
    let y_ordered = code.compare(Comparison::Eq, y, y);
    code.select(y_ordered, result, nan)
}
