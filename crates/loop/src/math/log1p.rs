//! `log1p`, as fdlibm corrects it: `log(u) + c`, where `u = 1 + x` rounded
//! and `c = ((1 + x) - u) / u` is that rounding's error, which `u` and `x`
//! give exactly.

use super::builder::ProgramBuilder;
use super::log::log_of;
use super::program::{Comparison, FloatSlot, MathProgram};

pub(super) fn log1p() -> MathProgram {
    let (mut code, x) = ProgramBuilder::new();
    let one = code.float(1.0);
    let u = code.add(one, x);
    let correction = correction(&mut code, x, u);
    let log = log_of(&mut code, u);
    let sum = code.add(log, correction);
    let result = specials(&mut code, x, sum);
    code.finish(result)
}

/// `((1 + x) - u) / u`. Below 2, `u - 1` is exact; from 2, `u - x` is.
fn correction(code: &mut ProgramBuilder, x: FloatSlot, u: FloatSlot) -> FloatSlot {
    let one = code.float(1.0);
    let u_less_x = code.sub(u, x);
    let from_x = code.sub(one, u_less_x);
    let u_less_one = code.sub(u, one);
    let from_one = code.sub(x, u_less_one);
    let two = code.float(2.0);
    let large = code.compare(Comparison::Ge, u, two);
    let error = code.select(large, from_x, from_one);
    code.div(error, u)
}

/// `sum` with the cases its correction gets wrong: `log1p(-1) = -inf`,
/// `log1p(+inf) = +inf`, and `log1p(±0) = ±0`.
fn specials(code: &mut ProgramBuilder, x: FloatSlot, sum: FloatSlot) -> FloatSlot {
    let minus_one = code.float(-1.0);
    let is_minus_one = code.compare(Comparison::Eq, x, minus_one);
    let minus_infinity = code.float(f32::NEG_INFINITY);
    let result = code.select(is_minus_one, minus_infinity, sum);
    let infinity = code.float(f32::INFINITY);
    let is_infinite = code.compare(Comparison::Eq, x, infinity);
    let result = code.select(is_infinite, infinity, result);
    let zero = code.float(0.0);
    let is_zero = code.compare(Comparison::Eq, x, zero);
    code.select(is_zero, x, result)
}
