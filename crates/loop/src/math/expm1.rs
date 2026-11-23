//! `expm1`: exp's reduction `x = k ln 2 + r`, then
//! `e^x - 1 = 2^k s + (2^k - 1)` with `s = e^r - 1`, which rounds once
//! while `2^k - 1` is exact; past that, `e^x - 1` from exp's scale, where
//! the `- 1` is too small to cancel.

use super::builder::ProgramBuilder;
use super::exp::{Reduced, clamp, expm1_reduced, power_of_two, reduce, scale};
use super::program::{Comparison, FloatSlot, MathProgram};

/// Below this, `e^x - 1` rounds to -1, and `2^k` stays a normal f32.
const LOW: f32 = -30.0;

/// Above this, `e^x - 1` is beyond the largest finite f32.
const HIGH: f32 = 89.0;

/// The largest `k` for which `2^k - 1` is an exact f32.
const EXACT_BELOW_POWER: f32 = 24.0;

pub(super) fn expm1() -> MathProgram {
    let (mut code, x) = ProgramBuilder::new();
    let result = expm1_of(&mut code, x);
    code.finish(result)
}

/// `e^x - 1`, signed zeros kept.
pub(super) fn expm1_of(code: &mut ProgramBuilder, argument: FloatSlot) -> FloatSlot {
    let x = clamp(code, argument, (LOW, HIGH));
    let Reduced { k, r } = reduce(code, x);
    let s = expm1_reduced(code, r);
    let moderate = moderate(code, s, k);
    let large = large(code, s, k);
    let limit = code.float(EXACT_BELOW_POWER);
    let is_large = code.compare(Comparison::Gt, k, limit);
    let result = code.select(is_large, large, moderate);
    let zero = code.float(0.0);
    let is_zero = code.compare(Comparison::Eq, argument, zero);
    code.select(is_zero, argument, result)
}

/// `2^k s + (2^k - 1)`, rounded once, for `k <= 24`.
fn moderate(code: &mut ProgramBuilder, s: FloatSlot, k: FloatSlot) -> FloatSlot {
    let n = code.float_to_int(k);
    let power = power_of_two(code, n);
    let one = code.float(1.0);
    let power_less_one = code.sub(power, one);
    code.fma(power, s, power_less_one)
}

/// `(1 + s) 2^k - 1`.
fn large(code: &mut ProgramBuilder, s: FloatSlot, k: FloatSlot) -> FloatSlot {
    let one = code.float(1.0);
    let p = code.add(one, s);
    let exp = scale(code, p, k);
    code.sub(exp, one)
}
