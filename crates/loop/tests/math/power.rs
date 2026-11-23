//! `pow` against f64 `powf` over sampled pairs, and C's special cases
//! against f32 `powf` exactly.

use crate::accuracy::ulps;
use ptxc_loop::math;

const SAMPLE_SEED: u64 = 0x90F5_EED5;
const SAMPLE_COUNT: usize = 2_000_000;

/// The largest error over 100 million pairs from each sampler (1.457
/// ulps), rounded up.
const BOUND_ULPS: f64 = 1.5;

fn pow(x: f32, y: f32) -> f32 {
    math::pow().eval(x, y)
}

/// xorshift64 pairs: `x` of either sign with an exponent in `[-40, 40]`,
/// and `y` either a fraction in `[-40, 40]` or an integer in `[-40, 40]`.
fn sample_pairs(seed: u64, count: usize) -> impl Iterator<Item = (f32, f32)> {
    let mut state = seed;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    std::iter::repeat_with(move || {
        let bits = next();
        let exponent = u32::try_from((bits >> 32) % 81).unwrap() + 127 - 40;
        let mantissa = u32::try_from(bits & 0x007F_FFFF).unwrap();
        let sign = u32::try_from((bits >> 40) & 1).unwrap() << 31;
        let x = f32::from_bits(sign | exponent << 23 | mantissa);
        let choice = next();
        let unit = f32::from_bits(0x3F80_0000 | u32::try_from(choice & 0x007F_FFFF).unwrap()) - 1.0;
        let y = if choice >> 63 == 0 {
            unit.mul_add(80.0, -40.0)
        } else {
            f32::from(i8::try_from((choice >> 32) % 81).unwrap()) - 40.0
        };
        (x, y)
    })
    .take(count)
}

#[test]
fn sampled_pairs_are_within_the_bound() {
    let worst = sample_pairs(SAMPLE_SEED, SAMPLE_COUNT)
        .map(|(x, y)| (ulps(pow(x, y), f64::from(x).powf(f64::from(y))), x, y))
        .fold((0.0, 0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a });
    eprintln!("pow: {:.3} ulps at ({:e}, {:e})", worst.0, worst.1, worst.2);
    assert!(worst.0 <= BOUND_ULPS, "pow is {} ulps off at ({:e}, {:e})", worst.0, worst.1, worst.2);
}

#[test]
fn special_cases_match_c() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        3.0,
        -3.0,
        2.5,
        -2.5,
        f32::from_bits(1),
        16_777_216.0,
        -16_777_217.0,
        3e9,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for x in values {
        for y in values {
            let (actual, expected) = (pow(x, y), x.powf(y));
            let close = ulps(actual, f64::from(x).powf(f64::from(y))) <= BOUND_ULPS;
            let same_special = if expected.is_finite() && expected != 0.0 && expected.abs() != 1.0 {
                close && actual.is_sign_negative() == expected.is_sign_negative()
            } else {
                actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan())
            };
            assert!(same_special, "pow({x:e}, {y:e}) = {actual:e}, expected {expected:e}");
        }
    }
}

/// Bases within 4096 ulps of 1 and exponents up to `2^20`, where
/// `log2 x` is tiny and `y` magnifies its error the most.
#[test]
fn bases_near_one_with_large_exponents_are_within_the_bound() {
    let mut state = SAMPLE_SEED;
    let mut worst = (0.0, 0.0, 0.0);
    for _ in 0..SAMPLE_COUNT {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let offset = u32::try_from(state % 4096).unwrap() + 1;
        let one = 1.0_f32.to_bits();
        let x = f32::from_bits(if state >> 63 == 0 { one + offset } else { one - offset });
        let unit =
            f32::from_bits(0x3F80_0000 | u32::try_from((state >> 20) & 0x007F_FFFF).unwrap()) - 1.0;
        let y = unit.mul_add(2_097_152.0, -1_048_576.0);
        let error = ulps(pow(x, y), f64::from(x).powf(f64::from(y)));
        if error > worst.0 {
            worst = (error, x, y);
        }
    }
    eprintln!("pow near 1: {:.3} ulps at ({:e}, {:e})", worst.0, worst.1, worst.2);
    assert!(worst.0 <= BOUND_ULPS, "pow is {} ulps off at ({:e}, {:e})", worst.0, worst.1, worst.2);
}
