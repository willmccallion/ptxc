//! Integers rounded by the encoder round once, as their exact value would.
//! Below 2^53 an integer is an exact f64, so it must encode as that f64
//! does. Above, the integer rounded to odd at 53 bits is an exact f64 that
//! every format at least two bits narrower rounds to the same encoding.

use crate::codec::{ALL, BF16_CODEC};

/// The bits an f64 holds exactly.
const F64_PRECISION: u32 = 53;

const SAMPLE_SEED: u64 = 0x1D7E_6E25;
const SAMPLE_COUNT: usize = 100_000;

/// `x` truncated to its top 53 significant bits, with the lowest kept bit
/// set when truncation dropped any: an exact f64.
#[expect(clippy::cast_precision_loss, reason = "the value has at most 53 significant bits")]
fn rounded_to_odd(x: i64) -> f64 {
    let magnitude = x.unsigned_abs();
    let width = magnitude.bit_width();
    let dropped = width.saturating_sub(F64_PRECISION);
    let kept = magnitude >> dropped << dropped;
    let sticky = if kept == magnitude { 0 } else { 1 << dropped };
    let odd = (kept | sticky) as f64;
    if x < 0 { -odd } else { odd }
}

/// xorshift64 over every bit pattern, so magnitudes of every width appear.
fn sample(seed: u64, count: usize) -> impl Iterator<Item = i64> {
    let mut state = seed;
    std::iter::repeat_with(move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let shift = u32::try_from(state % 64).unwrap();
        i64::from_ne_bytes((state >> shift).to_ne_bytes())
    })
    .take(count)
}

#[test]
#[expect(clippy::cast_precision_loss, reason = "every value here is below 2^53")]
fn small_integers_encode_as_their_exact_f64() {
    for codec in ALL {
        for x in -70_000..=70_000 {
            assert_eq!((codec.encode_int)(x), (codec.encode_f64)(x as f64), "{} {x}", codec.name);
        }
    }
}

#[test]
fn sampled_integers_round_once() {
    for codec in ALL {
        for x in sample(SAMPLE_SEED, SAMPLE_COUNT).chain([i64::MIN, i64::MAX, -1, 0, 1]) {
            let expected = (codec.encode_f64)(rounded_to_odd(x));
            assert_eq!((codec.encode_int)(x), expected, "{} {x}", codec.name);
        }
    }
}

/// 2^62 + 2^54 + 1 is just above the midpoint between two bf16 values, but
/// its nearest f64 is the midpoint itself, which ties to even.
#[test]
fn an_integer_just_above_a_tie_rounds_up() {
    let x = (1_i64 << 62) + (1 << 54) + 1;
    let upper = (BF16_CODEC.encode_f64)(2f64.powi(62) + 2f64.powi(55));

    assert_eq!((BF16_CODEC.encode_int)(x), upper);
    assert_eq!((BF16_CODEC.encode_int)(-x), upper | BF16_CODEC.sign_bit());
}
