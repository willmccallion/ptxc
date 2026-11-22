//! One description of a narrow binary float format, and the encoder and
//! decoder every format shares.
//!
//! The encoder works on the exact value as an `f64`, so `f32` inputs and
//! `f64` accumulators are both rounded once.

// Bit-field arithmetic on float encodings: every narrowing below is bounded
// by the format's field widths, which are at most 16 bits in total.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "bit-field arithmetic bounded by the format's field widths"
)]

/// What a finite value too large for the format, or an infinity, becomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overflow {
    /// The largest finite value of the same sign, as ONNX `Cast` with
    /// `saturate = 1`.
    Saturate,
    /// Infinity where the format has one and NaN where it does not, as IEEE
    /// conversion and ONNX `Cast` with `saturate = 0`.
    NonFinite,
}

/// How a format spends its all-ones exponent field.
#[derive(Clone, Copy, Debug)]
enum Specials {
    /// Infinities and NaNs, as IEEE 754.
    Ieee,
    /// Normal values, except all-ones mantissa which is NaN; no infinities
    /// (the "FN" formats, such as E4M3FN).
    NanOnly,
}

/// A binary float format with a sign bit, a biased exponent field and an
/// implicit leading one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FloatFormat {
    exponent_bits: u32,
    mantissa_bits: u32,
    bias: i32,
    specials: Specials,
}

/// The bits an `f64` spends on its fraction.
const F64_FRACTION_BITS: u32 = 52;

/// The exponent of an `f64`'s least significant fraction bit when its
/// biased exponent field is 1 or 0.
const F64_MIN_QUANTUM_EXPONENT: i32 = -1074;

impl FloatFormat {
    /// IEEE 754 binary16.
    pub(crate) const F16: Self =
        Self { exponent_bits: 5, mantissa_bits: 10, bias: 15, specials: Specials::Ieee };

    /// bfloat16: the top half of an IEEE binary32.
    pub(crate) const BF16: Self =
        Self { exponent_bits: 8, mantissa_bits: 7, bias: 127, specials: Specials::Ieee };

    /// FP8 E4M3FN, as in the OCP 8-bit float specification and ONNX.
    pub(crate) const E4M3FN: Self =
        Self { exponent_bits: 4, mantissa_bits: 3, bias: 7, specials: Specials::NanOnly };

    /// FP8 E5M2, as in the OCP 8-bit float specification and ONNX.
    pub(crate) const E5M2: Self =
        Self { exponent_bits: 5, mantissa_bits: 2, bias: 15, specials: Specials::Ieee };

    /// Encodes `value`, rounding to nearest with ties to even.
    pub(crate) fn encode(self, value: f64, overflow: Overflow) -> u16 {
        let sign = if value.is_sign_negative() { self.sign_bit() } else { 0 };
        if value.is_nan() {
            return sign | self.nan();
        }
        let magnitude = value.abs();
        let rounded = if magnitude.is_infinite() { None } else { self.round_magnitude(magnitude) };
        sign | rounded.unwrap_or_else(|| self.overflowed(overflow))
    }

    /// Decodes an encoding to its exact value.
    pub(crate) fn decode(self, bits: u16) -> f64 {
        let exponent_field = u32::from(bits >> self.mantissa_bits) & self.exponent_all_ones();
        let mantissa = u32::from(bits) & self.mantissa_all_ones();
        let magnitude = self.decode_magnitude(exponent_field, mantissa);
        if bits & self.sign_bit() == 0 { magnitude } else { -magnitude }
    }

    /// Decodes an encoding to an `f32`, which holds every value of every
    /// format exactly.
    pub(crate) fn decode_f32(self, bits: u16) -> f32 {
        self.decode(bits) as f32
    }

    /// Rounds a finite non-negative value to the format's grid; `None` when
    /// the rounded value is beyond the largest finite one.
    fn round_magnitude(self, magnitude: f64) -> Option<u16> {
        let (significand, exponent) = decompose(magnitude);
        if significand == 0 {
            return Some(0);
        }
        let leading = exponent + significand.ilog2() as i32;
        let quantum = leading.max(self.min_exponent()) - self.mantissa_bits as i32;
        let rounded = shift_right_ties_even(significand, (quantum - exponent) as u32);
        self.pack_rounded(rounded, quantum)
    }

    /// Packs `rounded * 2^quantum`, where `rounded` has at most one bit more
    /// than the mantissa holds.
    const fn pack_rounded(self, rounded: u64, quantum: i32) -> Option<u16> {
        let implicit = 1_u64 << self.mantissa_bits;
        let (rounded, quantum) =
            if rounded == implicit << 1 { (implicit, quantum + 1) } else { (rounded, quantum) };
        if rounded < implicit {
            return Some(self.pack(0, rounded as u32));
        }
        let exponent_field = quantum + self.mantissa_bits as i32 + self.bias;
        let mantissa = (rounded - implicit) as u32;
        let (max_exponent_field, max_mantissa) = self.largest_finite();
        let exceeds = exponent_field > max_exponent_field as i32
            || (exponent_field == max_exponent_field as i32 && mantissa > max_mantissa);
        if exceeds { None } else { Some(self.pack(exponent_field as u32, mantissa)) }
    }

    fn decode_magnitude(self, exponent_field: u32, mantissa: u32) -> f64 {
        let all_ones = exponent_field == self.exponent_all_ones();
        match self.specials {
            Specials::Ieee if all_ones && mantissa == 0 => f64::INFINITY,
            Specials::Ieee if all_ones => f64::NAN,
            Specials::NanOnly if all_ones && mantissa == self.mantissa_all_ones() => f64::NAN,
            _ if exponent_field == 0 => {
                f64::from(mantissa) * pow2(self.min_exponent() - self.mantissa_bits as i32)
            }
            _ => {
                let significand = (1 << self.mantissa_bits) | mantissa;
                let quantum = exponent_field as i32 - self.bias - self.mantissa_bits as i32;
                f64::from(significand) * pow2(quantum)
            }
        }
    }

    const fn overflowed(self, overflow: Overflow) -> u16 {
        match (overflow, self.specials) {
            (Overflow::Saturate, _) => {
                let (exponent_field, mantissa) = self.largest_finite();
                self.pack(exponent_field, mantissa)
            }
            (Overflow::NonFinite, Specials::Ieee) => self.pack(self.exponent_all_ones(), 0),
            (Overflow::NonFinite, Specials::NanOnly) => self.nan(),
        }
    }

    /// The exponent field and mantissa of the largest finite value.
    const fn largest_finite(self) -> (u32, u32) {
        match self.specials {
            Specials::Ieee => (self.exponent_all_ones() - 1, self.mantissa_all_ones()),
            Specials::NanOnly => (self.exponent_all_ones(), self.mantissa_all_ones() - 1),
        }
    }

    /// A positive quiet NaN.
    const fn nan(self) -> u16 {
        match self.specials {
            Specials::Ieee => self.pack(self.exponent_all_ones(), 1 << (self.mantissa_bits - 1)),
            Specials::NanOnly => self.pack(self.exponent_all_ones(), self.mantissa_all_ones()),
        }
    }

    const fn pack(self, exponent_field: u32, mantissa: u32) -> u16 {
        ((exponent_field << self.mantissa_bits) | mantissa) as u16
    }

    const fn sign_bit(self) -> u16 {
        1 << (self.exponent_bits + self.mantissa_bits)
    }

    const fn exponent_all_ones(self) -> u32 {
        (1 << self.exponent_bits) - 1
    }

    const fn mantissa_all_ones(self) -> u32 {
        (1 << self.mantissa_bits) - 1
    }

    /// The exponent of the smallest normal value.
    const fn min_exponent(self) -> i32 {
        1 - self.bias
    }
}

/// Splits a finite non-negative `f64` into `significand * 2^exponent`.
const fn decompose(magnitude: f64) -> (u64, i32) {
    let bits = magnitude.to_bits();
    let exponent_field = (bits >> F64_FRACTION_BITS) as i32;
    let fraction = bits & ((1 << F64_FRACTION_BITS) - 1);
    if exponent_field == 0 {
        (fraction, F64_MIN_QUANTUM_EXPONENT)
    } else {
        (fraction | (1 << F64_FRACTION_BITS), exponent_field - 1 + F64_MIN_QUANTUM_EXPONENT)
    }
}

/// `value / 2^shift`, rounded to nearest with ties to even.
const fn shift_right_ties_even(value: u64, shift: u32) -> u64 {
    if shift == 0 {
        return value;
    }
    if shift >= u64::BITS {
        // value < 2^53, below half of 2^shift: rounds to zero.
        return 0;
    }
    let kept = value >> shift;
    let dropped = value & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    if dropped > half || (dropped == half && kept & 1 == 1) { kept + 1 } else { kept }
}

/// `2^exponent` for an exponent in the normal `f64` range.
const fn pow2(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023) as u64) << F64_FRACTION_BITS)
}
