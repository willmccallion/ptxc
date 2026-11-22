//! Each format and overflow mode under test, behind one interface.

use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16, Overflow};

/// A format with a fixed overflow mode, as seen through the public API.
pub(crate) struct Codec {
    pub(crate) name: &'static str,
    pub(crate) width_bits: u32,
    pub(crate) encode: fn(f32) -> u16,
    pub(crate) decode: fn(u16) -> f64,
    pub(crate) decode_f32: fn(u16) -> f32,
    /// What a positive overflow encodes to in this mode.
    pub(crate) overflow_bits: u16,
}

impl Codec {
    pub(crate) const fn sign_bit(&self) -> u16 {
        1 << (self.width_bits - 1)
    }
}

pub(crate) const F16_CODEC: Codec = Codec {
    name: "f16",
    width_bits: 16,
    encode: |x| F16::from_f32(x).to_bits(),
    decode: |bits| F16::from_bits(bits).to_f64(),
    decode_f32: |bits| F16::from_bits(bits).to_f32(),
    overflow_bits: 0x7C00,
};

pub(crate) const BF16_CODEC: Codec = Codec {
    name: "bf16",
    width_bits: 16,
    encode: |x| Bf16::from_f32(x).to_bits(),
    decode: |bits| Bf16::from_bits(bits).to_f64(),
    decode_f32: |bits| Bf16::from_bits(bits).to_f32(),
    overflow_bits: 0x7F80,
};

pub(crate) const E4M3FN_SATURATE: Codec = Codec {
    name: "e4m3fn saturate",
    width_bits: 8,
    encode: |x| F8E4M3Fn::from_f32(x, Overflow::Saturate).to_bits().into(),
    decode: decode_e4m3fn,
    decode_f32: decode_e4m3fn_f32,
    overflow_bits: 0x7E,
};

pub(crate) const E4M3FN_NON_FINITE: Codec = Codec {
    name: "e4m3fn non-finite",
    width_bits: 8,
    encode: |x| F8E4M3Fn::from_f32(x, Overflow::NonFinite).to_bits().into(),
    decode: decode_e4m3fn,
    decode_f32: decode_e4m3fn_f32,
    overflow_bits: 0x7F,
};

pub(crate) const E5M2_SATURATE: Codec = Codec {
    name: "e5m2 saturate",
    width_bits: 8,
    encode: |x| F8E5M2::from_f32(x, Overflow::Saturate).to_bits().into(),
    decode: decode_e5m2,
    decode_f32: decode_e5m2_f32,
    overflow_bits: 0x7B,
};

pub(crate) const E5M2_NON_FINITE: Codec = Codec {
    name: "e5m2 non-finite",
    width_bits: 8,
    encode: |x| F8E5M2::from_f32(x, Overflow::NonFinite).to_bits().into(),
    decode: decode_e5m2,
    decode_f32: decode_e5m2_f32,
    overflow_bits: 0x7C,
};

pub(crate) const ALL: [&Codec; 6] = [
    &F16_CODEC,
    &BF16_CODEC,
    &E4M3FN_SATURATE,
    &E4M3FN_NON_FINITE,
    &E5M2_SATURATE,
    &E5M2_NON_FINITE,
];

fn decode_e4m3fn(bits: u16) -> f64 {
    F8E4M3Fn::from_bits(u8::try_from(bits).unwrap()).to_f64()
}

fn decode_e5m2(bits: u16) -> f64 {
    F8E5M2::from_bits(u8::try_from(bits).unwrap()).to_f64()
}

fn decode_e4m3fn_f32(bits: u16) -> f32 {
    F8E4M3Fn::from_bits(u8::try_from(bits).unwrap()).to_f32()
}

fn decode_e5m2_f32(bits: u16) -> f32 {
    F8E5M2::from_bits(u8::try_from(bits).unwrap()).to_f32()
}
