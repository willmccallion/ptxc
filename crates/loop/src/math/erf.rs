//! `erf`, odd, on `|x|` in two ranges: `x P(x^2)` below 0.875, and
//! `1 - e^Q(x)` above, where `Q` approximates `log(erfc x)`, which is
//! smooth where `erfc` itself falls by seven orders of magnitude.
//!
//! The coefficients are mpmath's `chebyfit` at 50 digits, rounded to f32:
//! `P` of `erf(sqrt t) / sqrt t` on `[0, 0.875^2]`, degree 6, error
//! 2.1e-10; `Q` of `log(erfc(u + 0.875))` on `[0, 3.125]`, degree 10,
//! error 4.3e-9.

use super::builder::ProgramBuilder;
use super::exp::exp_of;
use super::program::{Comparison, FloatSlot, IntOp, MathProgram};
use ptxc_core::convert::f64_to_f32;

/// Where the two ranges meet.
const BOUNDARY: f32 = 0.875;

/// Beyond this magnitude, `erf x` rounds to ±1 (from 3.9192).
const SATURATION: f32 = 4.0;

/// The f32 sign bit.
const SIGN_BIT: i32 = i32::MIN;

/// `P`'s terms above the constant, `2 / sqrt(pi)`, lowest degree first.
const SMALL: [f32; 6] = [
    -0.376_126_36,
    0.112_837_35,
    -0.026_861_706,
    0.005_207_049,
    -0.000_821_457_1,
    0.000_086_903_69,
];

/// `Q` in `u = x - BOUNDARY`: its constant term, then the rest, lowest
/// degree first. Measured from the boundary, the sum does not cancel where
/// `erfc` is largest and its error weighs most.
const LARGE: (f32, [f32; 10]) = (
    -1.532_824_4,
    [
        -2.430_220_1,
        -0.826_537_24,
        -0.046_919_115,
        0.011_385_848,
        -0.002_245_350_2,
        0.000_306_682_7,
        -0.000_015_040_088,
        -0.000_003_783_846_8,
        0.000_000_836_189_6,
        -0.000_000_054_926_06,
    ],
);

pub(super) fn erf() -> MathProgram {
    let (mut code, x) = ProgramBuilder::new();
    let bits = code.bits(x);
    let magnitude_mask = code.int(!SIGN_BIT);
    let magnitude_bits = code.int_op(IntOp::And, bits, magnitude_mask);
    let magnitude = code.float_from_bits(magnitude_bits);
    let limit = code.float(SATURATION);
    let saturates = code.compare(Comparison::Gt, magnitude, limit);
    let magnitude = code.select(saturates, limit, magnitude);
    let small = small(&mut code, magnitude);
    let large = large(&mut code, magnitude);
    let boundary = code.float(BOUNDARY);
    let is_small = code.compare(Comparison::Lt, magnitude, boundary);
    let unsigned = code.select(is_small, small, large);
    let unsigned_bits = code.bits(unsigned);
    let sign_mask = code.int(SIGN_BIT);
    let sign = code.int_op(IntOp::And, bits, sign_mask);
    let signed = code.int_op(IntOp::Or, unsigned_bits, sign);
    let result = code.float_from_bits(signed);
    code.finish(result)
}

/// `x P(x^2)`, as `x + x (P(x^2) - 1)` rounded once, so `P`'s rounding
/// errors weigh against `P - 1`, about an eighth of `P`.
fn small(code: &mut ProgramBuilder, x: FloatSlot) -> FloatSlot {
    let t = code.mul(x, x);
    let constant = f64_to_f32(std::f64::consts::FRAC_2_SQRT_PI - 1.0);
    let excess = code.polynomial(t, constant, &SMALL);
    code.fma(x, excess, x)
}

/// `1 - e^Q(x)`.
fn large(code: &mut ProgramBuilder, x: FloatSlot) -> FloatSlot {
    let boundary = code.float(BOUNDARY);
    let u = code.sub(x, boundary);
    let q = code.polynomial(u, LARGE.0, &LARGE.1);
    let erfc = exp_of(code, q);
    let one = code.float(1.0);
    code.sub(one, erfc)
}
