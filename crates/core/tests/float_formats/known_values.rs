//! Encodings taken from the format definitions, including each tie and
//! overflow boundary.

use crate::codec::{
    BF16_CODEC, Codec, E4M3FN_NON_FINITE, E4M3FN_SATURATE, E5M2_NON_FINITE, E5M2_SATURATE,
    F16_CODEC,
};

fn assert_encodes(codec: &Codec, cases: &[(f32, u16)]) {
    for &(input, bits) in cases {
        assert_eq!((codec.encode)(input), bits, "{} encoding {input:e}", codec.name);
    }
}

fn pow2(exponent: i32) -> f32 {
    2.0_f32.powi(exponent)
}

#[test]
fn f16_encodes_ieee_binary16_values() {
    assert_encodes(
        &F16_CODEC,
        &[
            (1.0, 0x3C00),
            (-2.0, 0xC000),
            (-0.0, 0x8000),
            (65504.0, 0x7BFF),
            (65519.0, 0x7BFF),
            (65520.0, 0x7C00),
            (pow2(-24), 0x0001),
            (pow2(-25), 0x0000),
            (1.5 * pow2(-25), 0x0001),
            (pow2(-14), 0x0400),
            (f32::INFINITY, 0x7C00),
            (f32::NEG_INFINITY, 0xFC00),
        ],
    );
}

#[test]
fn bf16_rounds_the_low_half_to_nearest_even() {
    assert_encodes(
        &BF16_CODEC,
        &[
            (1.0, 0x3F80),
            (f32::from_bits(0x3F80_8000), 0x3F80),
            (f32::from_bits(0x3F81_8000), 0x3F82),
            (f32::from_bits(0x3F80_8001), 0x3F81),
            (f32::MAX, 0x7F80),
            (f32::from_bits(0x0000_0001), 0x0000),
            (f32::from_bits(0x0001_0000), 0x0001),
        ],
    );
}

#[test]
fn e4m3fn_ties_to_448_and_overflows_past_it() {
    let shared = [
        (1.0, 0x38),
        (448.0, 0x7E),
        (464.0, 0x7E),
        (pow2(-9), 0x01),
        (pow2(-10), 0x00),
        (-0.0, 0x80),
    ];
    assert_encodes(&E4M3FN_SATURATE, &shared);
    assert_encodes(&E4M3FN_NON_FINITE, &shared);
    assert_encodes(
        &E4M3FN_SATURATE,
        &[(465.0, 0x7E), (f32::INFINITY, 0x7E), (f32::NEG_INFINITY, 0xFE)],
    );
    assert_encodes(
        &E4M3FN_NON_FINITE,
        &[(465.0, 0x7F), (f32::INFINITY, 0x7F), (f32::NEG_INFINITY, 0xFF)],
    );
}

#[test]
fn e5m2_ties_past_57344_overflow() {
    let shared = [(1.0, 0x3C), (57344.0, 0x7B), (pow2(-16), 0x01), (pow2(-17), 0x00)];
    assert_encodes(&E5M2_SATURATE, &shared);
    assert_encodes(&E5M2_NON_FINITE, &shared);
    assert_encodes(&E5M2_SATURATE, &[(61440.0, 0x7B), (f32::INFINITY, 0x7B)]);
    assert_encodes(
        &E5M2_NON_FINITE,
        &[(61440.0, 0x7C), (f32::INFINITY, 0x7C), (f32::NEG_INFINITY, 0xFC)],
    );
}

#[test]
fn nan_encodes_to_nan_in_every_format() {
    for codec in crate::codec::ALL {
        let bits = (codec.encode)(f32::NAN);
        assert!((codec.decode)(bits).is_nan(), "{} encoded NaN as {bits:#x}", codec.name);
    }
}
