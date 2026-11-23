//! `sin` and `cos`: `|x|` below `π/4` is its own remainder, and above it
//! Payne and Hanek's reduction gives the remainder `r` and the quadrant.
//! Taylor polynomials give `sin r` and `cos r` for `|r| <= π/4`; the
//! quadrant picks one and its sign. `sin` is odd, so it also takes `x`'s
//! sign; `cos` is `sin` a quadrant later.

use super::builder::ProgramBuilder;
use super::exp::inverse_factorial;
use super::program::{Comparison, FloatSlot, IntOp, IntSlot, MathProgram};
use super::reduce::{Quadrant, reduce};

/// The f32 sign bit.
const SIGN_BIT: i32 = i32::MIN;

/// The highest odd power in `sin`'s series: the first term left out,
/// `(π/4)^11 / 11!`, is below `2^-28` of the result.
const SIN_DEGREE: u32 = 9;

/// The highest even power in `cos`'s series: the first term left out,
/// `(π/4)^12 / 12!`, is below `2^-33`.
const COS_DEGREE: u32 = 10;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Function {
    Sin,
    Cos,
}

pub(super) fn sin() -> MathProgram {
    sin_or_cos(Function::Sin)
}

pub(super) fn cos() -> MathProgram {
    sin_or_cos(Function::Cos)
}

fn sin_or_cos(function: Function) -> MathProgram {
    let (mut code, x) = ProgramBuilder::new();
    let bits = code.bits(x);
    let magnitude_mask = code.int(!SIGN_BIT);
    let magnitude_bits = code.int_op(IntOp::And, bits, magnitude_mask);
    let magnitude = code.float_from_bits(magnitude_bits);
    let Quadrant { quadrant, r } = reduce_any(&mut code, magnitude);
    let sine = sin_kernel(&mut code, r);
    let cosine = cos_kernel(&mut code, r);
    let shift = code.int(i32::from(function == Function::Cos));
    let shifted = code.int_op(IntOp::Add, quadrant, shift);
    let odd_mask = code.int(1);
    let odd = code.int_op(IntOp::And, shifted, odd_mask);
    let is_odd = code.compare_int(Comparison::Eq, odd, odd_mask);
    let value = code.select(is_odd, cosine, sine);
    let flip = negation(&mut code, shifted, (function == Function::Sin).then_some(bits));
    let value_bits = code.bits(value);
    let signed = code.int_op(IntOp::Xor, value_bits, flip);
    let result = code.float_from_bits(signed);
    let result = not_finite_to_nan(&mut code, magnitude, result);
    code.finish(result)
}

/// `|x|`'s quadrant and remainder: itself below `π/4`, else reduced.
fn reduce_any(code: &mut ProgramBuilder, magnitude: FloatSlot) -> Quadrant {
    let reduced = reduce(code, magnitude);
    let quarter_pi = code.float(std::f32::consts::FRAC_PI_4);
    let small = code.compare(Comparison::Lt, magnitude, quarter_pi);
    let zero = code.int(0);
    let quadrant = code.select_int(small, zero, reduced.quadrant);
    let r = code.select(small, magnitude, reduced.r);
    Quadrant { quadrant, r }
}

/// The sign bit to flip: set in quadrants 2 and 3, and xor'd with the
/// argument's sign bit where one is given.
fn negation(code: &mut ProgramBuilder, quadrant: IntSlot, argument: Option<IntSlot>) -> IntSlot {
    let thirty = code.int(30);
    let shifted = code.int_op(IntOp::Shl, quadrant, thirty);
    let sign_mask = code.int(SIGN_BIT);
    let flip = code.int_op(IntOp::And, shifted, sign_mask);
    let Some(bits) = argument else {
        return flip;
    };
    let sign = code.int_op(IntOp::And, bits, sign_mask);
    code.int_op(IntOp::Xor, flip, sign)
}

/// `sin r = r + r^3 P(r^2)`, rounded once at the end.
fn sin_kernel(code: &mut ProgramBuilder, r: FloatSlot) -> FloatSlot {
    let z = code.mul(r, r);
    let higher: Vec<f32> = (5..=SIN_DEGREE).step_by(2).map(signed_inverse_factorial).collect();
    let p = code.polynomial(z, signed_inverse_factorial(3), &higher);
    let cube = code.mul(r, z);
    code.fma(cube, p, r)
}

/// `cos r = 1 - z/2 + z^2 P(z)` with `z = r^2`, as fdlibm keeps the bits
/// `1 - z/2` rounds away: `w + (((1 - w) - z/2) + z^2 P(z))`.
fn cos_kernel(code: &mut ProgramBuilder, r: FloatSlot) -> FloatSlot {
    let z = code.mul(r, r);
    let higher: Vec<f32> = (6..=COS_DEGREE).step_by(2).map(signed_inverse_factorial).collect();
    let p = code.polynomial(z, signed_inverse_factorial(4), &higher);
    let half = code.float(0.5);
    let half_z = code.mul(half, z);
    let one = code.float(1.0);
    let w = code.sub(one, half_z);
    let lost = code.sub(one, w);
    let rounding = code.sub(lost, half_z);
    let square = code.mul(z, z);
    let tail = code.fma(square, p, rounding);
    code.add(w, tail)
}

/// `(-1)^(n/2) / n!` for the series' even `n` or `(-1)^((n-1)/2) / n!` for
/// odd: the sign of `r^n`'s term in `sin` or `cos`.
fn signed_inverse_factorial(n: u32) -> f32 {
    let magnitude = inverse_factorial(n);
    if (n / 2) % 2 == 1 { -magnitude } else { magnitude }
}

/// `result`, or NaN where `magnitude` is infinite or NaN.
fn not_finite_to_nan(
    code: &mut ProgramBuilder,
    magnitude: FloatSlot,
    result: FloatSlot,
) -> FloatSlot {
    let infinity = code.float(f32::INFINITY);
    let finite = code.compare(Comparison::Lt, magnitude, infinity);
    let nan = code.float(f32::NAN);
    code.select(finite, result, nan)
}
