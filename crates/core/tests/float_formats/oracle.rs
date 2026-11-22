//! An independent rounding oracle: the nearest representable value found by
//! searching a sorted table of every finite value, ties to the even
//! encoding. It shares no code with the encoder.

use crate::codec::Codec;

/// The encoding the oracle expects for a non-negative input.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Expected {
    Bits(u16),
    Overflow,
}

pub(crate) struct Oracle {
    /// Every finite non-negative value with its encoding, ascending.
    values: Vec<(f64, u16)>,
    /// The value one grid step past the largest finite one; rounding to it
    /// is an overflow.
    beyond_max: f64,
}

impl Oracle {
    pub(crate) fn new(codec: &Codec) -> Self {
        let mut values: Vec<(f64, u16)> = (0..codec.sign_bit())
            .map(|bits| ((codec.decode)(bits), bits))
            .filter(|(value, _)| value.is_finite())
            .collect();
        values.sort_by(|a, b| a.0.total_cmp(&b.0));
        let [.., (second_largest, _), (largest, _)] = values[..] else {
            panic!("{} has fewer than two finite values", codec.name);
        };
        Self { beyond_max: largest + (largest - second_largest), values }
    }

    /// Every exact midpoint between neighbouring values, including the one
    /// between the largest finite value and the next grid step.
    pub(crate) fn midpoints(&self) -> impl Iterator<Item = f64> + '_ {
        let finite = self.values.windows(2).map(|pair| f64::midpoint(pair[0].0, pair[1].0));
        let largest = self.values.last().map_or(0.0, |&(value, _)| value);
        finite.chain(std::iter::once(f64::midpoint(largest, self.beyond_max)))
    }

    /// The expected encoding of a finite or infinite non-negative `x`.
    pub(crate) fn expected(&self, x: f64) -> Expected {
        if x.is_infinite() {
            return Expected::Overflow;
        }
        let above = self.values.partition_point(|(value, _)| *value <= x);
        let (low, low_bits) = self.values[above - 1];
        if low == x {
            return Expected::Bits(low_bits);
        }
        let high = self.values.get(above).copied();
        let high_value = high.map_or(self.beyond_max, |(value, _)| value);
        let midpoint = f64::midpoint(low, high_value);
        let choose_low = x < midpoint || (x == midpoint && low_bits & 1 == 0);
        match (choose_low, high) {
            (true, _) => Expected::Bits(low_bits),
            (false, Some((_, high_bits))) => Expected::Bits(high_bits),
            (false, None) => Expected::Overflow,
        }
    }
}

/// Checks `x` and `-x` against the oracle; `x` must be non-negative and not
/// NaN.
pub(crate) fn check(codec: &Codec, oracle: &Oracle, x: f32) {
    let expected = match oracle.expected(x.into()) {
        Expected::Bits(bits) => bits,
        Expected::Overflow => codec.overflow_bits,
    };
    let actual = (codec.encode)(x);
    assert_eq!(actual, expected, "{} encoding {x:e} ({:#010x})", codec.name, x.to_bits());
    let negated = (codec.encode)(-x);
    assert_eq!(negated, expected | codec.sign_bit(), "{} encoding {:e}", codec.name, -x);
}
