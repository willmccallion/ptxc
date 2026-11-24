//! Where each buffer lives, element addresses, and loads and stores.

use super::values::{Held, Narrow, Stored, memory_type, stored};
use super::{CodegenError, Selector};
use crate::ast::{
    Address, B16, B32, B64, BinaryOp, Compare, F32, PtxType, Reg, Signedness, Space, Value,
};
use ptxc_core::DType;
use ptxc_loop::{BufferId, Expr, ExprId, MemScope};

/// Where a buffer's elements are.
#[derive(Clone, Debug)]
pub(super) enum Storage {
    /// A kernel argument: global memory at this address.
    Global(Reg<B64>),
    /// A declared shared or local array.
    Array { space: Space, name: String },
}

impl Selector<'_> {
    /// Loads every global buffer's address from its parameter.
    pub(super) fn parameters(&mut self) {
        for param in self.kernel.params() {
            let name = self.builder.param();
            let pointer =
                self.builder.load(Space::Param, PtxType::U64, Address::Array { name, offset: 0 });
            let global = self.builder.global_address(pointer);
            let _new = self.storage.insert(param.buffer, Storage::Global(global));
        }
    }

    /// Declares `buffer`, a shared or local buffer an `Alloc` brings into
    /// scope.
    pub(super) fn declare(&mut self, buffer: BufferId) -> Result<(), CodegenError> {
        let found = &self.kernel.buffers()[buffer];
        let space = match found.scope {
            MemScope::Shared => Space::Shared,
            MemScope::Local | MemScope::Global => Space::Local,
            MemScope::Fragment(_) => {
                return Err(CodegenError::Unsupported { op: "fragment statements, until #104" });
            }
        };
        let bytes = found.size.in_bytes(found.dtype).map_err(|_| CodegenError::TooLarge)?.get();
        let stem = if space == Space::Shared { "shared" } else { "local" };
        let name = self.builder.array(space, stem, bytes);
        let _new = self.storage.insert(buffer, Storage::Array { space, name });
        Ok(())
    }

    /// The address of element `index` of `buffer`, and its space, computed
    /// once a block.
    fn element(
        &mut self,
        buffer: BufferId,
        index: ExprId,
    ) -> Result<(Space, Address), CodegenError> {
        if let Some(at) = self.known.address(buffer, index) {
            return Ok(at);
        }
        let at = self.computed_element(buffer, index)?;
        self.known.remember_address(buffer, index, at.clone());
        Ok(at)
    }

    /// The address of element `index` of `buffer`, and its space. A
    /// constant index becomes the address's immediate offset.
    fn computed_element(
        &mut self,
        buffer: BufferId,
        index: ExprId,
    ) -> Result<(Space, Address), CodegenError> {
        let dtype = self.kernel.buffers()[buffer].dtype;
        let size = i64::try_from(dtype.element_size().get()).map_err(|_| CodegenError::TooLarge)?;
        let storage = self.storage.get(&buffer).ok_or(CodegenError::Unbound)?.clone();
        if let Expr::Const(position) = self.kernel.exprs().get(index) {
            let offset = position.checked_mul(size).ok_or(CodegenError::TooLarge)?;
            return Ok(match storage {
                Storage::Global(base) => (Space::Global, Address::Reg { base: base.id(), offset }),
                Storage::Array { space, name } => (space, Address::Array { name, offset }),
            });
        }
        let position = self.index(index)?;
        let scaled = self.builder.int(
            BinaryOp::Mul,
            Signedness::Signed,
            position.into(),
            Value::<B64>::int(size),
        );
        let (space, base) = match storage {
            Storage::Global(base) => (Space::Global, base),
            Storage::Array { space, name } => (space, self.builder.address_of(&name)),
        };
        let at = self.builder.int(BinaryOp::Add, Signedness::Signed, base.into(), scaled.into());
        Ok((space, Address::Reg { base: at.id(), offset: 0 }))
    }

    /// Element `index` of `buffer`, as a held value.
    pub(super) fn load(&mut self, buffer: BufferId, index: ExprId) -> Result<Held, CodegenError> {
        let dtype = self.kernel.buffers()[buffer].dtype;
        let (space, addr) = self.element(buffer, index)?;
        let ty = memory_type(dtype);
        Ok(match (dtype, Narrow::of(dtype)) {
            (DType::F32, _) => Held::Float(self.builder.load::<F32>(space, ty, addr)),
            (_, Some(kind)) => {
                let bits = self.builder.load::<B16>(space, ty, addr);
                Held::Float(self.widened(bits, kind))
            }
            (DType::I64, _) => Held::Int64(self.builder.load::<B64>(space, ty, addr)),
            (DType::Bool, _) => {
                let byte = self.builder.load::<B32>(space, ty, addr);
                let zero = Value::<B32>::int(0);
                Held::Bool(self.builder.compare_int(
                    Compare::Ne,
                    Signedness::Unsigned,
                    byte.into(),
                    zero,
                ))
            }
            _ => Held::Int32(self.builder.load::<B32>(space, ty, addr)),
        })
    }

    /// Stores `value` as element `index` of `buffer`.
    pub(super) fn store(
        &mut self,
        buffer: BufferId,
        index: ExprId,
        value: Held,
    ) -> Result<(), CodegenError> {
        let dtype = self.kernel.buffers()[buffer].dtype;
        let (space, addr) = self.element(buffer, index)?;
        let ty = memory_type(dtype);
        let form = self.narrow_form(value, dtype);
        match form.unwrap_or_else(|| stored(&mut self.builder, value, dtype)) {
            Stored::F32(value) => self.builder.store(space, ty, addr, value.into()),
            Stored::Half(value) => self.builder.store(space, ty, addr, value.into()),
            Stored::Int32(value) => self.builder.store(space, ty, addr, value.into()),
            Stored::Int64(value) => self.builder.store(space, ty, addr, value.into()),
        }
        Ok(())
    }

    /// `value`'s memory form if it was widened from `dtype`'s: storing it
    /// then needs no rounding, rounding a widened value being exact.
    fn narrow_form(&self, value: Held, dtype: DType) -> Option<Stored> {
        let Held::Float(wide) = value else { return None };
        let &(bits, kind) = self.narrow_forms.get(&wide)?;
        (Narrow::of(dtype) == Some(kind)).then_some(Stored::Half(bits))
    }
}
