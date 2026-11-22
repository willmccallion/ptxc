//! Every encoding of every format decodes to a value that encodes back to it.

use crate::codec::{ALL, BF16_CODEC, Codec, E4M3FN_SATURATE, E5M2_SATURATE, F16_CODEC};

fn all_encodings(codec: &Codec) -> impl Iterator<Item = u16> {
    (0..1_u32 << codec.width_bits).map(|bits| u16::try_from(bits).unwrap())
}

/// Saturating codecs turn infinity into the largest finite value by design.
fn keeps_infinity(codec: &Codec) -> bool {
    (codec.decode)(codec.overflow_bits).is_infinite()
}

#[test]
fn every_non_nan_encoding_round_trips() {
    for codec in ALL {
        for bits in all_encodings(codec) {
            let value = (codec.decode)(bits);
            if value.is_nan() || (value.is_infinite() && !keeps_infinity(codec)) {
                continue;
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "every narrow value is an exact f32"
            )]
            let narrowed = value as f32;
            assert_eq!((codec.encode)(narrowed), bits, "{} round trip of {bits:#x}", codec.name);
        }
    }
}

#[test]
fn nan_encoding_counts_match_the_format_definitions() {
    let expected = [
        (&F16_CODEC, 2 * 1023),
        (&BF16_CODEC, 2 * 127),
        (&E4M3FN_SATURATE, 2),
        (&E5M2_SATURATE, 2 * 3),
    ];
    for (codec, count) in expected {
        let nans = all_encodings(codec).filter(|&bits| (codec.decode)(bits).is_nan()).count();
        assert_eq!(nans, count, "{} NaN encodings", codec.name);
    }
}
