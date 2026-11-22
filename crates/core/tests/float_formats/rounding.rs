//! f32 inputs rounded by the encoder agree with the oracle: a seeded sample
//! always, and every f32 under `--ignored` (run it with `--release`).

use crate::codec::{ALL, Codec};
use crate::oracle::{Oracle, check};
use std::num::NonZeroUsize;
use std::thread;

const SAMPLE_SEED: u64 = 0x5EED_F10A;
const SAMPLE_COUNT: usize = 200_000;

/// The bit pattern of positive infinity: every non-negative non-NaN f32 lies
/// at or below it.
const POSITIVE_INFINITY_BITS: u32 = 0x7F80_0000;

/// xorshift64, enough to spread a sample over the f32 bit patterns.
fn sample_bits(seed: u64, count: usize) -> impl Iterator<Item = u32> {
    let mut state = seed;
    std::iter::repeat_with(move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        u32::try_from(state >> 32).unwrap() % (POSITIVE_INFINITY_BITS + 1)
    })
    .take(count)
}

#[test]
fn sampled_f32_inputs_round_to_nearest_even() {
    for codec in ALL {
        let oracle = Oracle::new(codec);
        for bits in sample_bits(SAMPLE_SEED, SAMPLE_COUNT) {
            check(codec, &oracle, f32::from_bits(bits));
        }
    }
}

#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "every midpoint of a narrow format is an exact f32"
)]
fn every_tie_rounds_to_the_even_neighbour() {
    for codec in ALL {
        let oracle = Oracle::new(codec);
        for midpoint in oracle.midpoints() {
            check(codec, &oracle, midpoint as f32);
        }
    }
}

#[test]
#[ignore = "exhaustive over every f32; run with --release -- --ignored"]
fn every_f32_input_rounds_to_nearest_even() {
    for codec in ALL {
        check_every_f32(codec);
    }
}

fn check_every_f32(codec: &Codec) {
    let oracle = Oracle::new(codec);
    let threads = thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let total = u64::from(POSITIVE_INFINITY_BITS) + 1;
    let chunk = total.div_ceil(u64::try_from(threads).unwrap());
    thread::scope(|scope| {
        for start in (0..total).step_by(usize::try_from(chunk).unwrap()) {
            let end = (start + chunk).min(total);
            let oracle = &oracle;
            let _worker = scope.spawn(move || {
                for bits in start..end {
                    check(codec, oracle, f32::from_bits(u32::try_from(bits).unwrap()));
                }
            });
        }
    });
}
