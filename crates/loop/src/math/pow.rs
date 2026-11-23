//! `pow`, as `2^(y log2 |x|)`. `y log2 |x|` reaches 150 before the result
//! overflows or underflows, so an f32 logarithm would cost tens of ulps;
//! `log2 |x|` and the product are instead carried as pairs of f32, a high
//! part and the low part rounding left out of it.

use super::builder::ProgramBuilder;
use super::constants::LN2_HI;
use super::exp::{clamp, expm1_reduced, scale};
use super::log::{TERMS, atanh_coefficient, decompose};
use super::pow_specials::with_specials;
use super::program::{Binary, Comparison, FloatSlot, IntOp, MathProgram};
use ptxc_core::convert::f64_to_f32;

/// Below this, `2^t` underflows to zero; above the next, it overflows.
/// Both keep the scale's two halves normal powers of two.
const EXPONENT_RANGE: (f32, f32) = (-160.0, 140.0);

/// The f32 sign bit.
const SIGN_BIT: i32 = i32::MIN;

/// `hi + lo`, with `lo` at most about an ulp of `hi`.
#[derive(Clone, Copy)]
struct Wide {
    hi: FloatSlot,
    lo: FloatSlot,
}

pub(super) fn pow() -> MathProgram<Binary> {
    let (mut code, x, y) = ProgramBuilder::new_binary();
    let bits = code.bits(x);
    let mask = code.int(!SIGN_BIT);
    let magnitude_bits = code.int_op(IntOp::And, bits, mask);
    let magnitude = code.float_from_bits(magnitude_bits);
    let log2 = wide_log2(&mut code, magnitude);
    let t = times(&mut code, y, log2);
    let power = exp2(&mut code, t);
    let result = with_specials(&mut code, (x, y), power);
    code.finish(result)
}

/// `log2 a` for a positive finite `a = 2^e m`, `a` being `magnitude`: `e + log(m) log2(e)`, with
/// `log m = 2 atanh(s)` and `s = f / (2 + f)`, `f = m - 1`, as `log` does.
fn wide_log2(code: &mut ProgramBuilder, magnitude: FloatSlot) -> Wide {
    let (m, e) = decompose(code, magnitude);
    let one = code.float(1.0);
    let f = code.sub(m, one);
    let s = quotient(code, f);
    let square = code.mul(s.hi, s.hi);
    let higher: Vec<f32> = (2..=TERMS).map(atanh_coefficient).collect();
    let series = code.polynomial(square, atanh_coefficient(1), &higher);
    let tail = code.mul(square, series);
    let two = code.float(2.0);
    let twice_lo = code.mul(two, s.lo);
    let small = code.fma(s.hi, tail, twice_lo);
    let twice_hi = code.mul(two, s.hi);
    let log = fast_two_sum(code, twice_hi, small);
    let log2 = times_log2_e(code, log);
    let e = code.int_to_float(e);
    let sum = fast_two_sum(code, e, log2.hi);
    let lo = code.add(sum.lo, log2.lo);
    Wide { hi: sum.hi, lo }
}

/// `f / (2 + f)`, with `2 + f` itself split exactly, for `|f| < 1/2`.
fn quotient(code: &mut ProgramBuilder, f: FloatSlot) -> Wide {
    let two = code.float(2.0);
    let d_hi = code.add(two, f);
    let d_hi_less_two = code.sub(d_hi, two);
    let d_lo = code.sub(f, d_hi_less_two);
    let q_hi = code.div(f, d_hi);
    let minus_q = code.negate(q_hi);
    let residual = code.fma(minus_q, d_hi, f);
    let residual = code.fma(minus_q, d_lo, residual);
    let q_lo = code.div(residual, d_hi);
    Wide { hi: q_hi, lo: q_lo }
}

/// `a + b` as a pair, exact for `|a| >= |b|` or `a = 0`.
fn fast_two_sum(code: &mut ProgramBuilder, a: FloatSlot, b: FloatSlot) -> Wide {
    let hi = code.add(a, b);
    let a_less_hi = code.sub(a, hi);
    let lo = code.add(a_less_hi, b);
    Wide { hi, lo }
}

/// `v log2(e)`, with `log2(e)` split into its f32 and the rest.
fn times_log2_e(code: &mut ProgramBuilder, v: Wide) -> Wide {
    let constant = std::f32::consts::LOG2_E;
    let rest = f64_to_f32(std::f64::consts::LOG2_E - f64::from(constant));
    let (c_hi, c_lo) = (code.float(constant), code.float(rest));
    let hi = code.mul(v.hi, c_hi);
    let minus_hi = code.negate(hi);
    let error = code.fma(v.hi, c_hi, minus_hi);
    let cross = code.mul(v.lo, c_hi);
    let cross = code.fma(v.hi, c_lo, cross);
    let lo = code.add(error, cross);
    fast_two_sum(code, hi, lo)
}

/// `y v`, keeping the rounding error of `y v.hi`.
fn times(code: &mut ProgramBuilder, y: FloatSlot, v: Wide) -> Wide {
    let hi = code.mul(y, v.hi);
    let minus_hi = code.negate(hi);
    let error = code.fma(y, v.hi, minus_hi);
    let lo = code.fma(y, v.lo, error);
    Wide { hi, lo }
}

/// `2^exponent`, its low part dropped once its high part is clamped,
/// where the result overflows or underflows anyway: `exponent = k + remainder` with `k` integral, then `e^(remainder ln 2)` by exp's
/// polynomial and its two-step scale by `2^k`.
fn exp2(code: &mut ProgramBuilder, exponent: Wide) -> FloatSlot {
    let hi = clamp(code, exponent.hi, EXPONENT_RANGE);
    let in_range = code.compare(Comparison::Eq, hi, exponent.hi);
    let zero = code.float(0.0);
    let lo = code.select(in_range, exponent.lo, zero);
    let k = code.round_to_integer(hi);
    let fraction = code.sub(hi, k);
    let remainder = code.add(fraction, lo);
    let ln2_rest = f64_to_f32(std::f64::consts::LN_2 - f64::from(LN2_HI));
    let (ln2_hi, ln2_lo) = (code.float(LN2_HI), code.float(ln2_rest));
    let low = code.mul(remainder, ln2_lo);
    let r = code.fma(remainder, ln2_hi, low);
    let s = expm1_reduced(code, r);
    let one = code.float(1.0);
    let p = code.add(one, s);
    scale(code, p, k)
}
