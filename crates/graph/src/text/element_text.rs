//! Each element type's text form and the bytes its content hash reads.

use super::attr::{format_f32, parse_f32, push_formatted};
use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16, Overflow};

/// The 64-bit FNV-1a hash.
pub(super) struct Fnv64(u64);

impl Fnv64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    pub(super) const fn new() -> Self {
        Self(Self::OFFSET_BASIS)
    }

    pub(super) fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        }
    }

    pub(super) const fn finish(&self) -> u64 {
        self.0
    }
}

/// An element type's text form and bytes.
pub(super) trait ElementText: Copy {
    fn write(self, out: &mut String);
    fn parse(text: &str) -> Option<Self>;
    fn hash(self, hash: &mut Fnv64);
}

impl ElementText for f32 {
    fn write(self, out: &mut String) {
        out.push_str(&format_f32(self));
    }

    fn parse(text: &str) -> Option<Self> {
        parse_f32(text)
    }

    fn hash(self, hash: &mut Fnv64) {
        hash.write(&self.to_bits().to_le_bytes());
    }
}

/// Text for a narrow float: its exact value as an `f32`, or `nan_<bits>` in
/// its own encoding so NaN payloads survive.
macro_rules! narrow_float_text {
    ($($type:ty, $bits:ty, $encode:expr);+ $(;)?) => {$(
        impl ElementText for $type {
            fn write(self, out: &mut String) {
                let value = self.to_f32();
                if value.is_nan() {
                    push_formatted(out, format_args!("nan_{:x}", self.to_bits()));
                } else {
                    out.push_str(&format_f32(value));
                }
            }

            fn parse(text: &str) -> Option<Self> {
                if let Some(bits) = text.strip_prefix("nan_") {
                    let decoded = Self::from_bits(<$bits>::from_str_radix(bits, 16).ok()?);
                    return decoded.to_f32().is_nan().then_some(decoded);
                }
                let value: f32 = text.parse().ok()?;
                let encoded: Self = $encode(value);
                (encoded.to_f32().to_bits() == value.to_bits()).then_some(encoded)
            }

            fn hash(self, hash: &mut Fnv64) {
                hash.write(&self.to_bits().to_le_bytes());
            }
        }
    )+};
}

narrow_float_text! {
    F16, u16, F16::from_f32;
    Bf16, u16, Bf16::from_f32;
    F8E4M3Fn, u8, |value| F8E4M3Fn::from_f32(value, Overflow::NonFinite);
    F8E5M2, u8, |value| F8E5M2::from_f32(value, Overflow::NonFinite);
}

/// Text for an integer: decimal.
macro_rules! integer_text {
    ($($type:ty),+) => {$(
        impl ElementText for $type {
            fn write(self, out: &mut String) {
                out.push_str(&self.to_string());
            }

            fn parse(text: &str) -> Option<Self> {
                text.parse().ok()
            }

            fn hash(self, hash: &mut Fnv64) {
                hash.write(&self.to_le_bytes());
            }
        }
    )+};
}

integer_text!(i8, u8, i32, i64);

impl ElementText for bool {
    fn write(self, out: &mut String) {
        out.push_str(if self { "true" } else { "false" });
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        }
    }

    fn hash(self, hash: &mut Fnv64) {
        hash.write(&[u8::from(self)]);
    }
}
