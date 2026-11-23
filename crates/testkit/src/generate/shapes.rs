//! The generator's randomness, and shapes favouring the edge cases:
//! extents 0 and 1, primes, and sizes either side of a power of two.

/// `SplitMix64`: small, and the same on every platform, so a seed reproduces
/// its case exactly.
pub(crate) struct Rng(u64);

/// Extents to draw from, extent 1 weighted up.
const EXTENTS: [u64; 15] = [0, 1, 1, 1, 2, 3, 4, 5, 7, 8, 13, 16, 17, 31, 33];

impl Rng {
    pub(crate) const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) const fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `0..bound`, for a positive `bound`.
    pub(crate) const fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    /// A position in `0..len`, for a positive `len`.
    pub(crate) fn index(&mut self, len: usize) -> usize {
        usize::try_from(self.below(u64::try_from(len).unwrap())).unwrap()
    }

    /// One of `items`, which must not be empty.
    pub(crate) fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.index(items.len())]
    }

    /// A whole number from `low` to `high` inclusive, for `low <= high`.
    pub(crate) fn between(&mut self, low: i64, high: i64) -> i64 {
        let span = u64::try_from(high - low).unwrap() + 1;
        low + i64::try_from(self.below(span)).unwrap()
    }

    /// True with probability `numerator / denominator`.
    pub(crate) const fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        self.below(denominator) < numerator
    }
}

/// An extent, mostly from `EXTENTS`.
pub(crate) fn extent(rng: &mut Rng) -> u64 {
    *rng.pick(&EXTENTS)
}

/// Extents for an axis count of `rank`.
pub(crate) fn dims(rng: &mut Rng, rank: u64) -> Vec<u64> {
    (0..rank).map(|_| extent(rng)).collect()
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn the_stream_matches_splitmix64s_reference_outputs() {
        let mut rng = Rng::new(0);

        let draws = [rng.next(), rng.next(), rng.next()];

        assert_eq!(draws, [0xE220_A839_7B1D_CDAF, 0x6E78_9E6A_A1B9_65F4, 0x06C4_5D18_8009_454F]);
    }

    #[test]
    fn a_chance_comes_true_about_as_often_as_it_says() {
        let mut rng = Rng::new(1);

        let hits = (0..4000).filter(|_| rng.chance(1, 4)).count();

        assert!((900..1100).contains(&hits), "{hits}");
    }
}
