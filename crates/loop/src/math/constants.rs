//! Constants the programs share: the f32 encoding's fields, and ln 2 split
//! so its multiples by an exponent are exact.

use ptxc_core::convert::f64_to_f32;

/// The f32 exponent bias.
pub(super) const BIAS: i32 = 127;

/// The f32 mantissa's width in bits.
pub(super) const MANTISSA_BITS: i32 = 23;

/// ln 2 with its low nine mantissa bits cleared, so `k * LN2_HI` is exact
/// for every `|k| < 2^9`, which covers every f32 exponent.
pub(super) const LN2_HI: f32 = f32::from_bits(std::f32::consts::LN_2.to_bits() & !0x1FF);

/// `ln 2 - LN2_HI`, to f32.
pub(super) fn ln2_lo() -> f32 {
    f64_to_f32(std::f64::consts::LN_2 - f64::from(LN2_HI))
}
