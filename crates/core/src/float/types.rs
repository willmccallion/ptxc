//! The narrow float element types: f16, bf16, FP8 E4M3FN and FP8 E5M2.

use crate::float::format::{FloatFormat, Overflow};
use std::fmt;

/// Declares a narrow float type: its storage, raw-bit access, exact decoding
/// and a `Debug` that shows the value.
macro_rules! narrow_float {
    ($(#[$doc:meta])* $name:ident($storage:ty), $format:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy)]
        pub struct $name($storage);

        impl $name {
            /// Wraps a raw encoding.
            pub const fn from_bits(bits: $storage) -> Self {
                Self(bits)
            }

            /// The raw encoding.
            pub const fn to_bits(self) -> $storage {
                self.0
            }

            /// The exact value.
            pub fn to_f64(self) -> f64 {
                $format.decode(self.0.into())
            }

            /// The exact value; every value of this format is an `f32`.
            pub fn to_f32(self) -> f32 {
                $format.decode_f32(self.0.into())
            }

            fn encode(value: f64, overflow: Overflow) -> Self {
                Self(<$storage as NarrowBits>::from_encoding($format.encode(value, overflow)))
            }

            fn encode_int(value: i64, overflow: Overflow) -> Self {
                Self(<$storage as NarrowBits>::from_encoding($format.encode_int(value, overflow)))
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.to_f32())
            }
        }
    };
}

/// Storage that holds a format's encoding.
trait NarrowBits {
    fn from_encoding(bits: u16) -> Self;
}

impl NarrowBits for u16 {
    fn from_encoding(bits: u16) -> Self {
        bits
    }
}

impl NarrowBits for u8 {
    /// The 8-bit formats never set the high byte.
    fn from_encoding(bits: u16) -> Self {
        let [low, _high] = bits.to_le_bytes();
        low
    }
}

narrow_float!(
    /// IEEE 754 binary16.
    F16(u16),
    FloatFormat::F16
);

narrow_float!(
    /// bfloat16: an IEEE binary32 with the low 16 mantissa bits removed.
    Bf16(u16),
    FloatFormat::BF16
);

narrow_float!(
    /// FP8 E4M3FN: 4 exponent bits, 3 mantissa bits, no infinities, one NaN
    /// per sign. Largest finite value 448.
    F8E4M3Fn(u8),
    FloatFormat::E4M3FN
);

narrow_float!(
    /// FP8 E5M2: 5 exponent bits, 2 mantissa bits, IEEE infinities and NaNs.
    /// Largest finite value 57344.
    F8E5M2(u8),
    FloatFormat::E5M2
);

/// Constructors for a format whose overflow is always infinity.
macro_rules! ieee_constructors {
    ($($name:ident),+) => {$(
        impl $name {
            /// Rounds to nearest, ties to even; overflow becomes infinity.
            pub fn from_f64(value: f64) -> Self {
                Self::encode(value, Overflow::NonFinite)
            }

            /// Rounds to nearest, ties to even; overflow becomes infinity.
            pub fn from_f32(value: f32) -> Self {
                Self::from_f64(value.into())
            }

            /// Rounds once to nearest, ties to even; overflow becomes
            /// infinity.
            pub fn from_i64(value: i64) -> Self {
                Self::encode_int(value, Overflow::NonFinite)
            }
        }
    )+};
}

/// Constructors for an FP8 format, whose overflow follows ONNX `Cast`'s
/// `saturate` attribute.
macro_rules! fp8_constructors {
    ($($name:ident),+) => {$(
        impl $name {
            /// Rounds to nearest, ties to even; out-of-range values and
            /// infinities follow `overflow`.
            pub fn from_f64(value: f64, overflow: Overflow) -> Self {
                Self::encode(value, overflow)
            }

            /// Rounds to nearest, ties to even; out-of-range values and
            /// infinities follow `overflow`.
            pub fn from_f32(value: f32, overflow: Overflow) -> Self {
                Self::from_f64(value.into(), overflow)
            }

            /// Rounds once to nearest, ties to even; out-of-range values
            /// follow `overflow`.
            pub fn from_i64(value: i64, overflow: Overflow) -> Self {
                Self::encode_int(value, overflow)
            }
        }
    )+};
}

ieee_constructors!(F16, Bf16);
fp8_constructors!(F8E4M3Fn, F8E5M2);
