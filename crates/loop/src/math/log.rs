//! `log`, as fdlibm computes it: `x = 2^e m` with `m` in
//! `[sqrt(1/2), sqrt 2)`, then `log(1 + f)` for `f = m - 1` as
//! `f - (f^2/2 - s (f^2/2 + R))`, where `s = f / (2 + f)` and `R` is the
//! tail of `log(1 + f) = 2 atanh(s)`'s series.

use super::builder::ProgramBuilder;
use super::constants::{BIAS, LN2_HI, MANTISSA_BITS, ln2_lo};
use super::program::{Comparison, FloatSlot, IntOp, IntSlot, MathProgram};
use ptxc_core::convert::f64_to_f32;

/// `2^23`, which scales a subnormal into the normal range.
const SUBNORMAL_SCALE: f32 = 8_388_608.0;

/// The exponent the subnormal scale adds.
const SUBNORMAL_SCALE_EXPONENT: i32 = 23;

/// The mantissa field of an f32.
const MANTISSA_MASK: i32 = 0x007F_FFFF;

/// The number of atanh series terms after `2s`: with `|s| <= 0.1716`, the
/// first left out, `2 s^13 / 13`, is below `2^-36` of the result.
const TERMS: i32 = 5;

/// `2 / (2n + 1)`, the series coefficient of `s^(2n+1)`, to f32.
fn atanh_coefficient(n: i32) -> f32 {
    f64_to_f32(2.0 / f64::from(2 * n + 1))
}

pub(super) fn log() -> MathProgram {
    let (mut code, x) = ProgramBuilder::new();
    let (m, e) = decompose(&mut code, x);
    let one = code.float(1.0);
    let f = code.sub(m, one);
    let e = code.int_to_float(e);
    let finite = log_reduced(&mut code, f, e);
    let result = specials(&mut code, x, finite);
    code.finish(result)
}

/// `m` and `e` with `x = 2^e m` and `m` in `[sqrt(1/2), sqrt 2)`, for a
/// positive finite `x`; a subnormal is scaled into the normal range first.
fn decompose(code: &mut ProgramBuilder, x: FloatSlot) -> (FloatSlot, IntSlot) {
    let min_normal = code.float(f32::MIN_POSITIVE);
    let tiny = code.compare(Comparison::Lt, x, min_normal);
    let factor = code.float(SUBNORMAL_SCALE);
    let scaled = code.mul(x, factor);
    let normal = code.select(tiny, scaled, x);
    let bits = code.bits(normal);
    let shift = code.int(MANTISSA_BITS);
    let field = code.int_op(IntOp::ShrLogical, bits, shift);
    let bias = code.int(BIAS);
    let unscaled = code.int_op(IntOp::Sub, field, bias);
    let (scale_exponent, zero) = (code.int(SUBNORMAL_SCALE_EXPONENT), code.int(0));
    let adjustment = code.select_int(tiny, scale_exponent, zero);
    let e = code.int_op(IntOp::Sub, unscaled, adjustment);
    let mask = code.int(MANTISSA_MASK);
    let mantissa = code.int_op(IntOp::And, bits, mask);
    let one = code.int(1.0_f32.to_bits().cast_signed());
    let m = code.int_op(IntOp::Or, mantissa, one);
    halve_above_sqrt2(code, m, e)
}

/// `(m / 2, e + 1)` if `m` exceeds `sqrt 2`, else `(m, e)`; `m` as bits.
fn halve_above_sqrt2(code: &mut ProgramBuilder, m: IntSlot, e: IntSlot) -> (FloatSlot, IntSlot) {
    let sqrt2 = code.int(std::f32::consts::SQRT_2.to_bits().cast_signed());
    let above = code.compare_int(Comparison::Gt, m, sqrt2);
    let exponent_one = code.int(1 << MANTISSA_BITS);
    let halved = code.int_op(IntOp::Sub, m, exponent_one);
    let m = code.select_int(above, halved, m);
    let one = code.int(1);
    let raised = code.int_op(IntOp::Add, e, one);
    let e = code.select_int(above, raised, e);
    (code.float_from_bits(m), e)
}

/// `e ln 2 + log(1 + f)`, for `|f| < sqrt 2 - 1`.
fn log_reduced(code: &mut ProgramBuilder, f: FloatSlot, e: FloatSlot) -> FloatSlot {
    let two = code.float(2.0);
    let denominator = code.add(two, f);
    let s = code.div(f, denominator);
    let z = code.mul(s, s);
    let higher: Vec<f32> = (2..=TERMS).map(atanh_coefficient).collect();
    let series = code.polynomial(z, atanh_coefficient(1), &higher);
    let tail = code.mul(z, series);
    let half = code.float(0.5);
    let square = code.mul(f, f);
    let half_square = code.mul(half, square);
    let sum = code.add(half_square, tail);
    let product = code.mul(s, sum);
    let lo = code.float(ln2_lo());
    let low = code.fma(e, lo, product);
    let difference = code.sub(half_square, low);
    let log1p = code.sub(f, difference);
    let hi = code.float(LN2_HI);
    code.fma(e, hi, log1p)
}

/// `finite` with IEEE's special cases: `log(+inf) = +inf`,
/// `log(±0) = -inf`, and NaN for a negative or NaN `x`.
fn specials(code: &mut ProgramBuilder, x: FloatSlot, finite: FloatSlot) -> FloatSlot {
    let infinity = code.float(f32::INFINITY);
    let is_infinite = code.compare(Comparison::Eq, x, infinity);
    let result = code.select(is_infinite, infinity, finite);
    let zero = code.float(0.0);
    let is_zero = code.compare(Comparison::Eq, x, zero);
    let minus_infinity = code.float(f32::NEG_INFINITY);
    let result = code.select(is_zero, minus_infinity, result);
    let is_non_negative = code.compare(Comparison::Ge, x, zero);
    let nan = code.float(f32::NAN);
    code.select(is_non_negative, result, nan)
}
