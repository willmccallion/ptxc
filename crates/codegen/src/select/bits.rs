//! The bits an integer register's value may have set, followed back through
//! the instructions that computed it, as LLVM's known-bits analysis does:
//! masks, shifts, multiplications by powers of two and sums of values whose
//! bits do not overlap stay exact; a sum that may carry may set any bit
//! from its lowest overlapping one up; anything else, a `Var` among them,
//! may set any bit. Arithmetic wraps, so a shift left loses what it moves
//! past the top.

use crate::ast::{BinaryOp, CvtMode, FunctionBuilder, Instr, Operand, PtxType, RegId, Special};
use ptxc_loop::LaunchShape;
use ptxc_target::{BlockCount, ThreadsPerBlock};

/// Every bit.
const ANY: u64 = u64::MAX;

/// The lanes of a warp.
const LANES: u64 = 32;

/// The bits the registers of a function being built may have, in a kernel
/// launched as `launch`.
pub(super) struct KnownBits<'a> {
    pub(super) builder: &'a FunctionBuilder,
    pub(super) launch: LaunchShape,
}

impl KnownBits<'_> {
    /// The bits `reg`'s value may have set.
    pub(super) fn possible_bits(&self, reg: RegId) -> u64 {
        let Some(instr) = self.builder.definition(reg) else { return ANY };
        match instr {
            Instr::Mov { src, .. } => self.operand_bits(*src),
            Instr::Special { special, .. } => below(self.special_extent(*special)),
            Instr::Cvt {
                mode: CvtMode::Plain,
                to: PtxType::S64 | PtxType::U64,
                from: PtxType::U32 | PtxType::U64,
                src,
                ..
            } => match src.as_slice() {
                [operand] => self.operand_bits(*operand),
                _ => ANY,
            },
            Instr::Binary { op, ty, a, b, .. } => self.binary_bits(*op, *ty, [*a, *b]),
            _ => ANY,
        }
    }

    fn operand_bits(&self, operand: Operand) -> u64 {
        match operand {
            Operand::Reg(reg) => self.possible_bits(reg),
            Operand::Int(value) => value.cast_unsigned(),
            Operand::F32(_) => ANY,
        }
    }

    fn binary_bits(&self, op: BinaryOp, ty: PtxType, [a, b]: [Operand; 2]) -> u64 {
        let (left, right) = (self.operand_bits(a), self.operand_bits(b));
        let shift = match b {
            Operand::Int(amount) => u32::try_from(amount).ok().filter(|&amount| amount < u64::BITS),
            Operand::Reg(_) | Operand::F32(_) => None,
        };
        match (op, shift) {
            (BinaryOp::And, _) => left & right,
            (BinaryOp::Or | BinaryOp::Xor, _) => left | right,
            (BinaryOp::Add, _) => sum_bits(left, right),
            (BinaryOp::Shl, Some(amount)) => left << amount,
            (BinaryOp::Mul, _) if matches!(b, Operand::Int(_)) && right.is_power_of_two() => {
                left << right.trailing_zeros()
            }
            (BinaryOp::Shr, Some(amount)) if !may_be_negative(ty, left) => left >> amount,
            _ => ANY,
        }
    }

    /// How many values `special` takes in the launch.
    fn special_extent(&self, special: Special) -> u64 {
        let threads = |axis: usize| u64::from(ThreadsPerBlock::get(self.launch.block[axis]));
        let blocks = |axis: usize| u64::from(BlockCount::get(self.launch.grid[axis]));
        match special {
            Special::ThreadX => threads(0),
            Special::ThreadY => threads(1),
            Special::ThreadZ => threads(2),
            Special::BlockX => blocks(0),
            Special::BlockY => blocks(1),
            Special::BlockZ => blocks(2),
            Special::LaneId => LANES,
        }
    }
}

/// The bits a value in `0..extent` may have set.
const fn below(extent: u64) -> u64 {
    if extent <= 1 { 0 } else { ANY >> (extent - 1).leading_zeros() }
}

/// The bits a sum of values with `a`'s and `b`'s bits may have: exactly
/// either's below the lowest bit both may have, and any from there up,
/// where a carry may reach.
const fn sum_bits(a: u64, b: u64) -> u64 {
    let both = a & b;
    if both == 0 { a | b } else { (a | b) | (ANY << both.trailing_zeros()) }
}

/// Whether a `ty` value with `bits` may be negative, so that shifting it
/// right fills its top bits with ones.
const fn may_be_negative(ty: PtxType, bits: u64) -> bool {
    let sign = match ty {
        PtxType::S8 => 7,
        PtxType::S32 => 31,
        PtxType::S64 => 63,
        _ => return false,
    };
    bits >> sign != 0
}

#[cfg(test)]
mod tests {
    use super::{ANY, KnownBits, below, may_be_negative, sum_bits};
    use crate::ast::{
        B32, B64, BinaryOp, CvtMode, FunctionBuilder, PtxType, Reg, Signedness, Special, Value,
    };
    use ptxc_loop::LaunchShape;
    use ptxc_target::{BlockCount, ThreadsPerBlock};

    /// A launch of one block of `threads` threads.
    fn block_of(threads: u32) -> LaunchShape {
        let one = ThreadsPerBlock::new(1);
        LaunchShape {
            grid: [BlockCount::new(1); 3],
            block: [ThreadsPerBlock::new(threads), one, one],
        }
    }

    fn widened(builder: &mut FunctionBuilder, special: Special) -> Reg<B64> {
        let value: Reg<B32> = builder.special(special);
        builder.convert(CvtMode::Plain, (PtxType::U64, PtxType::U32), value.into())
    }

    fn int(builder: &mut FunctionBuilder, op: BinaryOp, a: Reg<B64>, b: i64) -> Reg<B64> {
        builder.int(op, Signedness::Bits, a.into(), Value::<B64>::int(b))
    }

    #[test]
    fn a_lanes_coordinates_keep_exact_bits_through_masks_shifts_scales_and_sums() {
        let mut builder = FunctionBuilder::new("kernel");
        let lane = widened(&mut builder, Special::LaneId);
        let row = int(&mut builder, BinaryOp::And, lane, 7);
        let matrix = int(&mut builder, BinaryOp::Shr, lane, 3);
        let scaled = int(&mut builder, BinaryOp::Mul, matrix, 64);
        let shifted = int(&mut builder, BinaryOp::Shl, row, 3);
        let copied = builder.mov(PtxType::B64, Value::from(shifted));
        let sum = builder.int(BinaryOp::Add, Signedness::Signed, scaled.into(), copied.into());
        let flipped = int(&mut builder, BinaryOp::Xor, sum, 0x100);
        let toggled = int(&mut builder, BinaryOp::Xor, lane, 0x10);

        let known = KnownBits { builder: &builder, launch: block_of(128) };
        let bits = [lane, row, matrix, scaled, shifted, copied, sum, flipped, toggled]
            .map(|reg| known.possible_bits(reg.id()));

        assert_eq!(bits, [0x1f, 0x7, 0x3, 0xc0, 0x38, 0x38, 0xf8, 0x1f8, 0x1f]);
    }

    #[test]
    fn a_sum_that_may_carry_may_set_any_bit_from_its_lowest_shared_one() {
        let mut builder = FunctionBuilder::new("kernel");
        let thread = widened(&mut builder, Special::ThreadX);
        let next =
            builder.int(BinaryOp::Add, Signedness::Signed, thread.into(), Value::<B64>::int(4));

        let known = KnownBits { builder: &builder, launch: block_of(128) };

        assert_eq!(known.possible_bits(thread.id()), 0x7f);
        assert_eq!(known.possible_bits(next.id()), 0b11 | (ANY << 2));
    }

    #[test]
    fn a_variable_or_a_shift_past_the_width_may_set_any_bit() {
        let mut builder = FunctionBuilder::new("kernel");
        let lane = widened(&mut builder, Special::LaneId);
        let counter = builder.variable::<B64>();
        builder.assign(counter, PtxType::S64, Value::<B64>::int(0));
        let gone = int(&mut builder, BinaryOp::Shl, lane, 64);

        let known = KnownBits { builder: &builder, launch: block_of(32) };

        assert_eq!([counter.reg(), gone].map(|reg| known.possible_bits(reg.id())), [ANY; 2]);
    }

    #[test]
    fn a_product_by_other_than_a_power_of_two_or_a_shift_filling_ones_may_set_any_bit() {
        let mut builder = FunctionBuilder::new("kernel");
        let lane = widened(&mut builder, Special::LaneId);
        let tripled = int(&mut builder, BinaryOp::Mul, lane, 3);
        let unknown = builder.variable::<B64>();
        builder.assign(unknown, PtxType::S64, Value::<B64>::int(0));
        let signed = Signedness::Signed;
        let halved = builder.int(BinaryOp::Shr, signed, unknown.reg().into(), Value::<B64>::int(1));

        let known = KnownBits { builder: &builder, launch: block_of(32) };

        assert_eq!([tripled, halved].map(|reg| known.possible_bits(reg.id())), [ANY; 2]);
    }

    #[test]
    fn values_below_an_extent_set_only_the_bits_below_its_highest() {
        assert_eq!([below(0), below(1), below(2), below(8), below(9)], [0, 0, 1, 7, 15]);
    }

    #[test]
    fn a_sum_is_exact_until_its_addends_overlap_and_then_any_bit_above() {
        assert_eq!(sum_bits(0b1100, 0b0011), 0b1111);
        assert_eq!(sum_bits(0b0110, 0b0100), 0b0010 | (ANY << 2));
    }

    #[test]
    fn a_signed_value_may_be_negative_where_its_own_widths_sign_bit_may_be_set() {
        let bit = |n: u32| 1_u64 << n;
        assert!(may_be_negative(PtxType::S32, bit(31)));
        assert!(!may_be_negative(PtxType::S32, bit(30)));
        assert!(may_be_negative(PtxType::S64, bit(63)));
        assert!(!may_be_negative(PtxType::S64, bit(31)));
        assert!(may_be_negative(PtxType::S8, bit(7)));
        assert!(!may_be_negative(PtxType::U32, bit(31)));
    }
}
