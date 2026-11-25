//! Addresses in swizzled shared buffers (D66): each element's byte offset
//! where the buffer's swizzle places it, a constant addend folded in after
//! the rest is swizzled wherever `Swizzle::fold` allows, as CUTLASS's
//! iterators do (D67).

use super::bits::KnownBits;
use super::{CodegenError, Selector};
use crate::ast::{B64, BinaryOp, Reg, Signedness, Value};
use ptxc_loop::{BufferId, MemScope, Swizzle};

impl Selector<'_> {
    /// Where `buffer` places byte `bytes + constant`, as a register and an
    /// immediate to add to it.
    pub(super) fn placed_offset(
        &mut self,
        buffer: BufferId,
        bytes: Reg<B64>,
        constant: i64,
    ) -> Result<(Reg<B64>, i64), CodegenError> {
        let Some(swizzle) = self.swizzle_of(buffer) else { return Ok((bytes, constant)) };
        let known = KnownBits { builder: &self.builder, launch: self.kernel.launch() };
        let Some(fold) = swizzle.fold(known.possible_bits(bytes.id()), constant) else {
            let signed = Signedness::Signed;
            let sum =
                self.index_op(BinaryOp::Add, signed, bytes.into(), Value::<B64>::int(constant));
            return Ok((self.swizzled_offset(swizzle, sum), 0));
        };
        let swizzled = self.swizzled_offset(swizzle, bytes);
        if fold.xor == 0 {
            return Ok((swizzled, fold.add));
        }
        let xor = i64::try_from(fold.xor).map_err(|_| CodegenError::TooLarge)?;
        let bits = Signedness::Bits;
        let folded = self.index_op(BinaryOp::Xor, bits, swizzled.into(), Value::<B64>::int(xor));
        Ok((folded, fold.add))
    }

    /// Where `swizzle` places byte `offset`: a shift, a mask, a shift and
    /// an exclusive-or.
    fn swizzled_offset(&mut self, swizzle: Swizzle, offset: Reg<B64>) -> Reg<B64> {
        let bitwise = Signedness::Bits;
        let above = Value::<B64>::int(i64::from(swizzle.unit() + swizzle.shift()));
        let read = self.index_op(BinaryOp::Shr, bitwise, offset.into(), above);
        let mask = Value::<B64>::int((1 << swizzle.bits()) - 1);
        let bits = self.index_op(BinaryOp::And, bitwise, read.into(), mask);
        let unit = Value::<B64>::int(i64::from(swizzle.unit()));
        let moved = self.index_op(BinaryOp::Shl, bitwise, bits.into(), unit);
        self.index_op(BinaryOp::Xor, bitwise, offset.into(), moved.into())
    }

    /// Where `buffer`'s swizzle, if it has one, places constant byte
    /// `offset`.
    pub(super) fn swizzled_constant(
        &self,
        buffer: BufferId,
        offset: i64,
    ) -> Result<i64, CodegenError> {
        let Some(swizzle) = self.swizzle_of(buffer) else { return Ok(offset) };
        let unsigned = u64::try_from(offset).map_err(|_| CodegenError::Unbound)?;
        i64::try_from(swizzle.apply(unsigned)).map_err(|_| CodegenError::TooLarge)
    }

    /// `buffer`'s swizzle, if it is a swizzled shared buffer.
    pub(super) fn swizzle_of(&self, buffer: BufferId) -> Option<Swizzle> {
        match self.kernel.buffers()[buffer].scope {
            MemScope::Shared { swizzle } => swizzle,
            MemScope::Global | MemScope::Local | MemScope::Fragment(_) => None,
        }
    }
}
