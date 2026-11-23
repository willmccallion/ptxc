//! `tanh x = e / (e + 2)` with `e = expm1(2|x|)`, which keeps small
//! arguments' relative accuracy, and the sign copied back from `x`.

use super::builder::ProgramBuilder;
use super::expm1::expm1_of;
use super::program::{Comparison, IntOp, MathProgram};

/// Beyond this magnitude, `tanh x` rounds to ±1.
const SATURATION: f32 = 10.0;

/// The f32 sign bit.
const SIGN_BIT: i32 = i32::MIN;

pub(super) fn tanh() -> MathProgram {
    let (mut code, x) = ProgramBuilder::new();
    let bits = code.bits(x);
    let magnitude_mask = code.int(!SIGN_BIT);
    let magnitude_bits = code.int_op(IntOp::And, bits, magnitude_mask);
    let magnitude = code.float_from_bits(magnitude_bits);
    let limit = code.float(SATURATION);
    let saturates = code.compare(Comparison::Gt, magnitude, limit);
    let magnitude = code.select(saturates, limit, magnitude);
    let two = code.float(2.0);
    let doubled = code.mul(magnitude, two);
    let e = expm1_of(&mut code, doubled);
    let denominator = code.add(e, two);
    let quotient = code.div(e, denominator);
    let quotient_bits = code.bits(quotient);
    let sign_mask = code.int(SIGN_BIT);
    let sign = code.int_op(IntOp::And, bits, sign_mask);
    let signed = code.int_op(IntOp::Or, quotient_bits, sign);
    let result = code.float_from_bits(signed);
    code.finish(result)
}
