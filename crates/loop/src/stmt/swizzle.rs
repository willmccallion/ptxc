//! `Swizzle`: an XOR permutation of a shared buffer's bytes, as CUTLASS's
//! `Swizzle<B, M, S>`, so that the rows an `ldmatrix` or a copy touches at
//! once fall in different banks.

use ptxc_core::Dim;

/// A permutation of the units within each block of a buffer's bytes.
///
/// Byte offset `o` is stored at `o ^ (((o >> (unit + shift)) & (2^bits -
/// 1)) << unit)`: within each block of `2^(unit + bits)` bytes, the
/// `2^bits` units of `2^unit` bytes are permuted by the bits `shift` above
/// them. Since `shift` is at least `bits`, the bits it reads are ones it
/// leaves alone, so it is its own inverse and a permutation of every
/// block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Swizzle {
    bits: u32,
    unit: u32,
    shift: u32,
}

/// How a swizzle places `offset + constant`, from where it places `offset`:
/// exclusive-or `xor`, then add `add`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fold {
    /// The constant's part below the swizzle's period, placed.
    pub xor: u64,
    /// The constant's whole periods.
    pub add: i64,
}

/// The smallest unit a swizzle moves: 16 bytes, the widest copy and an
/// `ldmatrix` row, so neither straddles two units.
const MIN_UNIT: u32 = 4;

/// The largest offset bit a swizzle reads, below 32 so a shared address
/// holds it.
const MAX_BIT: u32 = 32;

/// The bits of a 128-byte line of the 32 four-byte banks.
const LINE_BITS: u32 = 7;

impl Swizzle {
    /// The swizzle permuting units of `2^unit` bytes, at least 16, by `bits`
    /// bits `shift` above them, if `shift` is at least `bits` and every bit
    /// it reads is below 32.
    pub const fn new(bits: u32, unit: u32, shift: u32) -> Option<Self> {
        let fits = match unit.checked_add(shift) {
            Some(start) => match start.checked_add(bits) {
                Some(end) => end <= MAX_BIT,
                None => false,
            },
            None => false,
        };
        if bits > 0 && unit >= MIN_UNIT && shift >= bits && fits {
            Some(Self { bits, unit, shift })
        } else {
            None
        }
    }

    /// The swizzle for a row-major tile whose rows are `row_bytes` long,
    /// as CUTLASS picks one: each 16-byte unit's place in its line
    /// exclusive-ored with the row's place among 8, so the 8 rows an
    /// `ldmatrix` reads fall in 8 different groups of banks; none for rows
    /// that are not a power of two of at least 16 bytes.
    pub fn for_rows(row_bytes: u64) -> Option<Self> {
        if row_bytes < (1 << MIN_UNIT) || !row_bytes.is_power_of_two() {
            return None;
        }
        // Rows shorter than the 128 bytes of a bank line share one, so the
        // bits above the line count the rows instead.
        let line_bits = row_bytes.trailing_zeros().max(LINE_BITS);
        Self::new(3, MIN_UNIT, line_bits - MIN_UNIT)
    }

    /// The swizzle `for_rows` picks for a row-major tile of `extents`
    /// elements of `element_bytes` each, if the tile holds whole blocks of
    /// it.
    pub fn for_tile(extents: &[Dim], element_bytes: u64) -> Option<Self> {
        let row = extents.last()?.get().checked_mul(element_bytes)?;
        let swizzle = Self::for_rows(row)?;
        let bytes = extents
            .iter()
            .try_fold(element_bytes, |bytes, extent| bytes.checked_mul(extent.get()))?;
        bytes.is_multiple_of(swizzle.block_bytes()).then_some(swizzle)
    }

    /// Where byte `offset` is stored.
    pub const fn apply(self, offset: u64) -> u64 {
        let mask = (1 << self.bits) - 1;
        offset ^ (((offset >> (self.unit + self.shift)) & mask) << self.unit)
    }

    /// The bytes of a block, which the swizzle permutes within.
    pub const fn block_bytes(self) -> u64 {
        1 << (self.unit + self.bits)
    }

    /// The bytes of the swizzle's period: adding a multiple of it to an
    /// offset adds the same to where the offset is placed.
    pub const fn period_bytes(self) -> u64 {
        1 << (self.unit + self.shift + self.bits)
    }

    /// How to place `offset + constant` from where `offset` is placed, for
    /// an `offset` that sets only bits among `possible`. If the constant's
    /// part below the period sets none of them, adding that part is an
    /// exclusive-or, which the swizzle keeps since it is linear over
    /// exclusive-or, and the whole periods it moves unchanged. `None`
    /// otherwise.
    pub const fn fold(self, possible: u64, constant: i64) -> Option<Fold> {
        let part = constant.rem_euclid(self.period_bytes().cast_signed());
        let bits = part.cast_unsigned();
        if possible & bits != 0 {
            return None;
        }
        Some(Fold { xor: self.apply(bits), add: constant - part })
    }

    /// The number of bits of a unit's place it changes.
    pub const fn bits(self) -> u32 {
        self.bits
    }

    /// The log2 of a unit's bytes.
    pub const fn unit(self) -> u32 {
        self.unit
    }

    /// How many bits above the unit's place the bits it reads are.
    pub const fn shift(self) -> u32 {
        self.shift
    }
}

#[cfg(test)]
mod tests {
    use super::Swizzle;
    use ptxc_core::Dim;
    use std::collections::HashSet;

    /// Every value that sets only bits of `mask`.
    fn within(mask: u64) -> Vec<u64> {
        let mut values = vec![0];
        let mut value = mask;
        while value != 0 {
            values.push(value);
            value = (value - 1) & mask;
        }
        values
    }

    #[test]
    fn a_constant_folds_in_exactly_when_its_part_below_the_period_misses_the_offsets_bits() {
        let swizzle = Swizzle::new(3, 4, 3).unwrap();
        // A row's units, the rows of a period, and rows of the next periods.
        let masks = [0x70, 0x380, 0x1c00, 0x3f0];

        for possible in masks {
            for constant in -2048..2048 {
                let fold = swizzle.fold(possible, constant);

                let part = constant.rem_euclid(1024).cast_unsigned();
                assert_eq!(fold.is_some(), possible & part == 0, "{possible:#x}, {constant}");
                let Some(fold) = fold else { continue };
                for offset in within(possible) {
                    let Some(total) = offset.checked_add_signed(constant) else { continue };
                    let folded = (swizzle.apply(offset) ^ fold.xor).wrapping_add_signed(fold.add);
                    assert_eq!(folded, swizzle.apply(total), "{offset:#x} + {constant}");
                }
            }
        }
    }

    #[test]
    fn a_swizzle_permutes_each_block_and_is_its_own_inverse() {
        let swizzle = Swizzle::new(3, 4, 3).unwrap();
        let offsets: Vec<u64> = (0..4096).collect();

        let moved: Vec<u64> = offsets.iter().map(|&offset| swizzle.apply(offset)).collect();

        let blocks = |offset: &u64| offset / swizzle.block_bytes();
        assert!(offsets.iter().zip(&moved).all(|(a, b)| blocks(a) == blocks(b)));
        assert_eq!(moved.iter().copied().collect::<HashSet<_>>().len(), offsets.len());
        assert!(moved.iter().zip(&offsets).all(|(&b, &a)| swizzle.apply(b) == a));
        assert_eq!(swizzle.apply(15), 15);
        assert_eq!(swizzle.apply(128), 128 + 16);
    }

    #[test]
    fn only_a_swizzle_reading_bits_it_leaves_alone_of_units_of_16_bytes_exists() {
        assert_eq!(Swizzle::new(3, 4, 2), None);
        assert_eq!(Swizzle::new(3, 3, 3), None);
        assert_eq!(Swizzle::new(0, 4, 3), None);
        assert_eq!(Swizzle::new(3, 4, 26), None);
        assert!(Swizzle::new(3, 4, 25).is_some());
    }

    #[test]
    fn rows_of_8_unit_places_each_read_by_8_rows_fall_in_8_bank_groups() {
        for row_bytes in [32_u64, 64, 128, 256, 512] {
            let swizzle = Swizzle::for_rows(row_bytes).unwrap();
            for unit in 0..row_bytes / 16 {
                let groups: HashSet<u64> = (0..8)
                    .map(|row| (swizzle.apply(row * row_bytes + unit * 16) % 128) / 16)
                    .collect();
                assert_eq!(groups.len(), 8, "{row_bytes}-byte rows, unit {unit}");
            }
        }
        assert_eq!([8, 48, 0].map(Swizzle::for_rows), [None, None, None]);
    }

    #[test]
    fn a_tile_takes_its_rows_swizzle_if_it_holds_whole_blocks_of_it() {
        let dims = |extents: &[u64]| extents.iter().copied().map(Dim::new).collect::<Vec<_>>();

        let wide = Swizzle::for_tile(&dims(&[8, 128]), 2);
        let narrow = Swizzle::for_tile(&dims(&[64, 32]), 2);
        let short = Swizzle::for_tile(&dims(&[2, 8]), 4);
        let ragged = Swizzle::for_tile(&dims(&[8, 40]), 4);

        assert_eq!(wide, Swizzle::new(3, 4, 4));
        assert_eq!(narrow, Swizzle::new(3, 4, 3));
        assert_eq!([short, ragged, Swizzle::for_tile(&[], 2)], [None, None, None]);
    }
}
