//! `fmod` is exact, so it must equal Rust's `%` on f32, bit for bit.

use ptxc_loop::math;

const SAMPLE_SEED: u64 = 0x0F0D_5EED;
const SAMPLE_COUNT: usize = 300_000;

fn assert_exact(x: f32, y: f32) {
    let (actual, expected) = (math::fmod().eval(x, y), x % y);
    let same = actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan());
    assert!(same, "fmod({x:e}, {y:e}) = {actual:e}, expected {expected:e}");
}

/// xorshift64, each state split into two f32 bit patterns.
fn sample_pairs(seed: u64, count: usize) -> impl Iterator<Item = (f32, f32)> {
    let mut state = seed;
    std::iter::repeat_with(move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let low = u32::try_from(state & u64::from(u32::MAX)).unwrap();
        let high = u32::try_from(state >> 32).unwrap();
        (f32::from_bits(low), f32::from_bits(high))
    })
    .take(count)
}

#[test]
fn sampled_pairs_match_rust_exactly() {
    for (x, y) in sample_pairs(SAMPLE_SEED, SAMPLE_COUNT) {
        assert_exact(x, y);
    }
}

#[test]
fn the_widest_exponent_gaps_and_edges_are_exact() {
    let tiny = f32::from_bits(1);
    let cases = [
        (f32::MAX, tiny),
        (f32::MAX, f32::from_bits(3)),
        (-f32::MAX, f32::MIN_POSITIVE),
        (f32::MAX, 0.1),
        (1e-40, 3e-44),
        (-5.5, 2.0),
        (5.5, -2.0),
        (-0.0, 1.0),
        (0.0, -1.0),
        (1.0, f32::INFINITY),
        (-3.0, f32::NEG_INFINITY),
        (f32::INFINITY, 1.0),
        (1.0, 0.0),
        (1.0, -0.0),
        (f32::NAN, 1.0),
        (1.0, f32::NAN),
        (f32::INFINITY, f32::INFINITY),
        (7.0, 7.0),
        (16_777_215.0, 0.5),
    ];
    for (x, y) in cases {
        assert_exact(x, y);
    }
}
