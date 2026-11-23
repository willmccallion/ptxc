//! Payne and Hanek's reduction of `|x|` modulo `π/2`, exact for every f32:
//! the 24-bit significand times a 96-bit window of `2/π`'s bits, chosen by
//! the exponent so the product's integer part, modulo 4, is the quadrant
//! and the 64 bits below it are the fraction of a quadrant.

use super::builder::ProgramBuilder;
use super::constants::{BIAS, MANTISSA_BITS};
use super::exp::power_of_two;
use super::program::{Comparison, FloatSlot, IntOp, IntSlot, PredSlot};
use ptxc_core::convert::f64_to_f32;

/// `2/π`'s first 224 bits after the point, as mpmath gives them and as
/// fdlibm's `ipio2` table starts, behind one zero word that stands for the
/// bits before the point.
pub(super) const TWO_OVER_PI: [u32; 8] =
    [0, 0xA2F9_836E, 0x4E44_1529, 0xFC27_57D1, 0xF534_DDC0, 0xDB62_9599, 0x3C43_9041, 0xFE51_63AB];

/// The table indices the window's first word can have.
const STARTS: [i32; 5] = [0, 1, 2, 3, 4];

/// The largest of `STARTS`.
const LAST_START: usize = 4;

/// The smallest exponent reduced; a smaller `|x|` is below `π/4` and is
/// its own remainder.
const LOWEST_EXPONENT: i32 = -1;

/// The largest finite exponent.
const HIGHEST_EXPONENT: i32 = 127;

/// The window starts this many bits past the table's start, plus the
/// exponent.
const WINDOW_OFFSET: i32 = 7;

/// The significand's shift, putting the quadrant in the product's bits
/// 96 and 97.
const SIGNIFICAND_SHIFT: i32 = 2;

/// `|x| = (4n + quadrant) π/2 + r`, with `|r| <= π/4`.
pub(super) struct Quadrant {
    pub(super) quadrant: IntSlot,
    pub(super) r: FloatSlot,
}

/// The quadrant and remainder of a finite `magnitude >= π/4`.
pub(super) fn reduce(code: &mut ProgramBuilder, magnitude: FloatSlot) -> Quadrant {
    let bits = code.bits(magnitude);
    let window = window(code, bits);
    let significand = significand(code, bits);
    let [p1, p2, p3] = product(code, significand, window);
    let three = code.int(3);
    let quadrant = code.int_op(IntOp::And, p3, three);
    let (quadrant, fraction, negative) = round_to_nearest(code, quadrant, [p1, p2]);
    let r = fraction_times_half_pi(code, fraction);
    let zero = code.float(0.0);
    let negated = code.sub(zero, r);
    Quadrant { quadrant, r: code.select(negative, negated, r) }
}

/// The 24-bit significand, shifted left by two.
fn significand(code: &mut ProgramBuilder, bits: IntSlot) -> IntSlot {
    let mask = code.int((1 << MANTISSA_BITS) - 1);
    let mantissa = code.int_op(IntOp::And, bits, mask);
    let implicit = code.int(1 << MANTISSA_BITS);
    let significand = code.int_op(IntOp::Or, mantissa, implicit);
    let shift = code.int(SIGNIFICAND_SHIFT);
    code.int_op(IntOp::Shl, significand, shift)
}

/// The 96 bits of `2/π` the exponent selects, most significant word first.
fn window(code: &mut ProgramBuilder, bits: IntSlot) -> [IntSlot; 3] {
    let shift = code.int(MANTISSA_BITS);
    let field = code.int_op(IntOp::ShrLogical, bits, shift);
    let bias = code.int(BIAS);
    let exponent = code.int_op(IntOp::Sub, field, bias);
    let exponent = clamp_int(code, exponent, (LOWEST_EXPONENT, HIGHEST_EXPONENT));
    let offset = code.int(WINDOW_OFFSET);
    let start = code.int_op(IntOp::Add, exponent, offset);
    let five = code.int(5);
    let index = code.int_op(IntOp::ShrLogical, start, five);
    let low_bits = code.int(31);
    let bit = code.int_op(IntOp::And, start, low_bits);
    let starts = STARTS.map(|n| {
        let n = code.int(n);
        code.compare_int(Comparison::Eq, index, n)
    });
    let words: Vec<IntSlot> = (0..4).map(|j| pick(code, &starts, j)).collect();
    let width = code.int(32);
    let complement = code.int_op(IntOp::Sub, width, bit);
    [0, 1, 2].map(|j| {
        let high = code.int_op(IntOp::Shl, words[j], bit);
        let low = code.int_op(IntOp::ShrLogical, words[j + 1], complement);
        code.int_op(IntOp::Or, high, low)
    })
}

/// The table word `offset` past the window's first, by a chain of selects
/// over the possible first words.
fn pick(code: &mut ProgramBuilder, starts: &[PredSlot], offset: usize) -> IntSlot {
    let word = |code: &mut ProgramBuilder, start: usize| {
        code.int(TWO_OVER_PI[start + offset].cast_signed())
    };
    let mut picked = word(code, LAST_START);
    for start in (0..LAST_START).rev() {
        let candidate = word(code, start);
        picked = code.select_int(starts[start], candidate, picked);
    }
    picked
}

/// `value` limited to `[low, high]`.
fn clamp_int(code: &mut ProgramBuilder, value: IntSlot, (low, high): (i32, i32)) -> IntSlot {
    let low = code.int(low);
    let below = code.compare_int(Comparison::Lt, value, low);
    let raised = code.select_int(below, low, value);
    let high = code.int(high);
    let above = code.compare_int(Comparison::Gt, raised, high);
    code.select_int(above, high, raised)
}

/// Bits 32 to 127 of the 128-bit `significand * window`, least significant
/// word first.
fn product(code: &mut ProgramBuilder, significand: IntSlot, window: [IntSlot; 3]) -> [IntSlot; 3] {
    let [w0, w1, w2] = window;
    let high2 = code.int_op(IntOp::MulHiUnsigned, significand, w2);
    let low1 = code.int_op(IntOp::Mul, significand, w1);
    let high1 = code.int_op(IntOp::MulHiUnsigned, significand, w1);
    let low0 = code.int_op(IntOp::Mul, significand, w0);
    let high0 = code.int_op(IntOp::MulHiUnsigned, significand, w0);
    let (p1, carry1) = add_with_carry(code, high2, low1);
    let (partial, carry_low) = add_with_carry(code, high1, low0);
    let (p2, carry_high) = add_with_carry(code, partial, carry1);
    let carry2 = code.int_op(IntOp::Add, carry_low, carry_high);
    let p3 = code.int_op(IntOp::Add, high0, carry2);
    [p1, p2, p3]
}

/// `a + b` modulo 2^32, and the carry out as 0 or 1.
fn add_with_carry(code: &mut ProgramBuilder, a: IntSlot, b: IntSlot) -> (IntSlot, IntSlot) {
    let sum = code.int_op(IntOp::Add, a, b);
    let wrapped = code.compare_unsigned(Comparison::Lt, sum, a);
    let (one, zero) = (code.int(1), code.int(0));
    (sum, code.select_int(wrapped, one, zero))
}

/// The quadrant and fraction rounded to the nearest quadrant: from half a
/// quadrant up, the next quadrant and the fraction's distance below it,
/// with a predicate saying the remainder is negative.
fn round_to_nearest(
    code: &mut ProgramBuilder,
    quadrant: IntSlot,
    [low, high]: [IntSlot; 2],
) -> (IntSlot, [IntSlot; 2], PredSlot) {
    let (zero, one) = (code.int(0), code.int(1));
    let top_bit = code.int(31);
    let half = code.int_op(IntOp::ShrLogical, high, top_bit);
    let negative = code.compare_int(Comparison::Eq, half, one);
    let next = code.int_op(IntOp::Add, quadrant, half);
    let three = code.int(3);
    let quadrant = code.int_op(IntOp::And, next, three);
    let negated_low = code.int_op(IntOp::Sub, zero, low);
    let borrows = code.compare_unsigned(Comparison::Gt, low, zero);
    let borrow = code.select_int(borrows, one, zero);
    let negated = code.int_op(IntOp::Sub, zero, high);
    let negated_high = code.int_op(IntOp::Sub, negated, borrow);
    let low = code.select_int(negative, negated_low, low);
    let high = code.select_int(negative, negated_high, high);
    (quadrant, [low, high], negative)
}

/// The 64-bit fraction `[low, high]` of a quadrant, times `π/2`: its
/// leading 32 bits, normalized, to the nearest f32, scaled by their place.
fn fraction_times_half_pi(code: &mut ProgramBuilder, [low, high]: [IntSlot; 2]) -> FloatSlot {
    let zero = code.int(0);
    let high_is_zero = code.compare_int(Comparison::Eq, high, zero);
    let high_zeros = code.leading_zeros(high);
    let low_zeros = code.leading_zeros(low);
    let width = code.int(32);
    let below_high = code.int_op(IntOp::Add, width, low_zeros);
    let zeros = code.select_int(high_is_zero, below_high, high_zeros);
    let shifted = code.int_op(IntOp::Shl, high, high_zeros);
    let rest = code.int_op(IntOp::Sub, width, high_zeros);
    let carried = code.int_op(IntOp::ShrLogical, low, rest);
    let from_high = code.int_op(IntOp::Or, shifted, carried);
    let from_low = code.int_op(IntOp::Shl, low, low_zeros);
    let top = code.select_int(high_is_zero, from_low, from_high);
    let leading = code.unsigned_to_float(top);
    let minus_width = code.int(-32);
    let exponent = code.int_op(IntOp::Sub, minus_width, zeros);
    let place = power_of_two(code, exponent);
    let fraction = code.mul(leading, place);
    let half_pi = f64_to_f32(std::f64::consts::FRAC_PI_2);
    let half_pi_lo = code.float(f64_to_f32(std::f64::consts::FRAC_PI_2 - f64::from(half_pi)));
    let tail = code.mul(fraction, half_pi_lo);
    let half_pi = code.float(half_pi);
    code.fma(fraction, half_pi, tail)
}
