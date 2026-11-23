//! `exp`, as Cody and Waite reduce it: `x = k ln 2 + r` with
//! `|r| <= ln 2 / 2`, a Taylor polynomial for `e^r`, and a scale by `2^k`
//! in two halves, so a subnormal result rounds once.

use super::builder::ProgramBuilder;
use super::constants::{BIAS, LN2_HI, MANTISSA_BITS, ln2_lo};
use super::program::{Comparison, FloatSlot, IntOp, IntSlot, MathProgram};
use ptxc_core::convert::f64_to_f32;

/// Below this, `e^x` is under half the least subnormal and rounds to zero.
const UNDERFLOW: f32 = -104.0;

/// Above this, `e^x` is beyond the largest finite f32.
const OVERFLOW: f32 = 89.0;

/// The Taylor polynomial's degree: the first term left out,
/// `(ln 2 / 2)^8 / 8!`, is below `2^-27`.
const DEGREE: u32 = 7;

/// `1 / n!`, to f32.
fn inverse_factorial(n: u32) -> f32 {
    f64_to_f32(1.0 / (1..=n).map(f64::from).product::<f64>())
}

pub(super) fn exp() -> MathProgram {
    let (mut code, argument) = ProgramBuilder::new();
    let x = clamp(&mut code, argument, (UNDERFLOW, OVERFLOW));
    let log2e = code.float(std::f32::consts::LOG2_E);
    let scaled = code.mul(x, log2e);
    let k = code.round_to_integer(scaled);
    let r = reduce(&mut code, x, k);
    let p = exp_reduced(&mut code, r);
    let result = scale(&mut code, p, k);
    code.finish(result)
}

/// `x` limited to `[low, high]`; NaN stays NaN.
fn clamp(code: &mut ProgramBuilder, x: FloatSlot, (low, high): (f32, f32)) -> FloatSlot {
    let low = code.float(low);
    let below = code.compare(Comparison::Lt, x, low);
    let raised = code.select(below, low, x);
    let high = code.float(high);
    let above = code.compare(Comparison::Gt, raised, high);
    code.select(above, high, raised)
}

/// `x - k ln 2`, in two fused steps.
fn reduce(code: &mut ProgramBuilder, x: FloatSlot, k: FloatSlot) -> FloatSlot {
    let minus_hi = code.float(-LN2_HI);
    let partial = code.fma(k, minus_hi, x);
    let minus_lo = code.float(-ln2_lo());
    code.fma(k, minus_lo, partial)
}

/// `e^r` for `|r| <= ln 2 / 2`, as `1 + (r + r^2 q(r))` with `q` the
/// Taylor series from its `r^2` term on.
fn exp_reduced(code: &mut ProgramBuilder, r: FloatSlot) -> FloatSlot {
    let higher: Vec<f32> = (3..=DEGREE).map(inverse_factorial).collect();
    let tail = code.polynomial(r, inverse_factorial(2), &higher);
    let square = code.mul(r, r);
    let sum = code.fma(square, tail, r);
    let one = code.float(1.0);
    code.add(one, sum)
}

/// `p 2^k` for an integral `k` in `[-150, 128]`, as two multiplications by
/// normal powers of two, of which only the second can round.
fn scale(code: &mut ProgramBuilder, p: FloatSlot, k: FloatSlot) -> FloatSlot {
    let k = code.float_to_int(k);
    let one = code.int(1);
    let half = code.int_op(IntOp::ShrArithmetic, k, one);
    let rest = code.int_op(IntOp::Sub, k, half);
    let first = power_of_two(code, half);
    let second = power_of_two(code, rest);
    let partial = code.mul(p, first);
    code.mul(partial, second)
}

/// `2^n` for `n` in the normal exponent range.
fn power_of_two(code: &mut ProgramBuilder, n: IntSlot) -> FloatSlot {
    let bias = code.int(BIAS);
    let biased = code.int_op(IntOp::Add, n, bias);
    let shift = code.int(MANTISSA_BITS);
    let bits = code.int_op(IntOp::Shl, biased, shift);
    code.float_from_bits(bits)
}
