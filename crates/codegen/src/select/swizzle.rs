//! Byte offsets into swizzled shared buffers: each where the buffer's
//! swizzle places it (D66), computed with a shift, a mask and an
//! exclusive-or, or at compile time for a constant.

use super::{CodegenError, Selector};
use crate::ast::{B64, BinaryOp, Reg, Signedness, Value};
use ptxc_loop::{BufferId, MemScope, Swizzle};

impl Selector<'_> {
    /// Where `buffer`'s swizzle, if it has one, places byte `offset`.
    pub(super) fn swizzled_offset(&mut self, buffer: BufferId, offset: Reg<B64>) -> Reg<B64> {
        let Some(swizzle) = self.swizzle_of(buffer) else { return offset };
        let bitwise = Signedness::Bits;
        let above = Value::<B64>::int(i64::from(swizzle.unit() + swizzle.shift()));
        let read = self.builder.int(BinaryOp::Shr, bitwise, offset.into(), above);
        let mask = Value::<B64>::int((1 << swizzle.bits()) - 1);
        let bits = self.builder.int(BinaryOp::And, bitwise, read.into(), mask);
        let unit = Value::<B64>::int(i64::from(swizzle.unit()));
        let moved = self.builder.int(BinaryOp::Shl, bitwise, bits.into(), unit);
        self.builder.int(BinaryOp::Xor, bitwise, offset.into(), moved.into())
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

    fn swizzle_of(&self, buffer: BufferId) -> Option<Swizzle> {
        match self.kernel.buffers()[buffer].scope {
            MemScope::Shared { swizzle } => swizzle,
            MemScope::Global | MemScope::Local | MemScope::Fragment(_) => None,
        }
    }
}
