//! Where each buffer lives, element addresses, and loads and stores.

use super::lanes::WARP_LANES;
use super::position::{Position, Ranked};
use super::registers::{WORD_BYTES, WordByte, register_place};
use super::values::{Held, Narrow, Stored, memory_type, stored};
use super::{CodegenError, Selector};
use crate::ast::{
    Address, B16, B32, B64, BinaryOp, Compare, F32, PtxType, Reg, Signedness, Space, Value, Var,
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
    /// A local or fragment array held in registers, 4 bytes a register.
    Registers(Vec<Var<B32>>),
}

/// Where an element is.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Place {
    /// Memory, in a space at an address.
    Memory(Space, Address),
    /// Bytes of a register holding 4 bytes of an array, from `byte`.
    Register { word: Var<B32>, byte: WordByte },
}

/// The place of the element `offset` bytes into `storage`.
pub(super) fn place_in(storage: &Storage, offset: i64) -> Result<Place, CodegenError> {
    Ok(match storage {
        Storage::Global(base) => {
            Place::Memory(Space::Global, Address::Reg { base: base.id(), offset })
        }
        Storage::Array { space, name } => {
            Place::Memory(*space, Address::Array { name: name.clone(), offset })
        }
        Storage::Registers(words) => register_place(words, offset)?,
    })
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
        let bytes = found.size.in_bytes(found.dtype).map_err(|_| CodegenError::TooLarge)?.get();
        let (space, bytes) = match found.scope {
            MemScope::Shared { .. } => (Space::Shared, bytes),
            MemScope::Local | MemScope::Global => (Space::Local, bytes),
            MemScope::Fragment(_) => (Space::Local, bytes / WARP_LANES),
        };
        let storage = if self.registers.contains(&buffer) {
            Storage::Registers(
                (0..bytes.div_ceil(WORD_BYTES)).map(|_| self.builder.variable()).collect(),
            )
        } else {
            let stem = if space == Space::Shared { "shared" } else { "local" };
            Storage::Array { space, name: self.builder.array(space, stem, bytes) }
        };
        let _new = self.storage.insert(buffer, storage);
        Ok(())
    }

    /// Where element `index` of `buffer` is, computed once a block.
    pub(super) fn element(
        &mut self,
        buffer: BufferId,
        index: ExprId,
    ) -> Result<Place, CodegenError> {
        if let Some(at) = self.known.address(buffer, index) {
            return Ok(at);
        }
        let at = self.computed_element(buffer, index)?;
        self.known.remember_address(buffer, index, at.clone());
        Ok(at)
    }

    /// The address of element `index` of `buffer`, which is in memory.
    pub(super) fn memory_element(
        &mut self,
        buffer: BufferId,
        index: ExprId,
    ) -> Result<(Space, Address), CodegenError> {
        match self.element(buffer, index)? {
            Place::Memory(space, addr) => Ok((space, addr)),
            Place::Register { .. } => {
                Err(CodegenError::Unsupported { op: "a vector access to registers" })
            }
        }
    }

    /// Where element `index` of `buffer` is. A constant index becomes the
    /// address's immediate offset, swizzled where the buffer is, or picks
    /// the register holding it.
    fn computed_element(&mut self, buffer: BufferId, index: ExprId) -> Result<Place, CodegenError> {
        let dtype = self.kernel.buffers()[buffer].dtype;
        let size = i64::try_from(dtype.element_size().get()).map_err(|_| CodegenError::TooLarge)?;
        if let Expr::Const(position) = self.kernel.exprs().get(index) {
            let offset = position.checked_mul(size).ok_or(CodegenError::TooLarge)?;
            let offset = self.swizzled_constant(buffer, offset)?;
            return place_in(self.storage.get(&buffer).ok_or(CodegenError::Unbound)?, offset);
        }
        let position = self.position(buffer, index)?;
        let (space, addr) = self.address_at(buffer, position)?;
        Ok(Place::Memory(space, addr))
    }

    /// The address of the element of `buffer` at `position`, and its
    /// space: the buffer's start plus each part in bytes, outermost rank
    /// first, and an immediate offset.
    pub(super) fn address_at(
        &mut self,
        buffer: BufferId,
        position: Position,
    ) -> Result<(Space, Address), CodegenError> {
        let space = self.space_of(buffer)?;
        let (at, immediate) = self.address_from(buffer, position)?;
        Ok((space, Address::Reg { base: at.id(), offset: immediate }))
    }

    /// The register `address_at` adds an immediate offset to, and the
    /// offset.
    pub(super) fn address_from(
        &mut self,
        buffer: BufferId,
        position: Position,
    ) -> Result<(Reg<B64>, i64), CodegenError> {
        let storage = self.storage.get(&buffer).ok_or(CodegenError::Unbound)?.clone();
        let (mut parts, immediate) = self.byte_parts(buffer, position)?;
        parts.sort_by_key(|part| part.rank);
        let mut at = match storage {
            Storage::Global(base) => base,
            Storage::Array { name, .. } => self.array_address(&name),
            Storage::Registers(_) => {
                return Err(CodegenError::Unsupported { op: "a computed index into registers" });
            }
        };
        for part in parts {
            at = self.added(at, part.value);
        }
        Ok((at, immediate))
    }

    /// The space `buffer`'s elements are in, if they are in memory.
    fn space_of(&self, buffer: BufferId) -> Result<Space, CodegenError> {
        match self.storage.get(&buffer).ok_or(CodegenError::Unbound)? {
            Storage::Global(_) => Ok(Space::Global),
            Storage::Array { space, .. } => Ok(*space),
            Storage::Registers(_) => {
                Err(CodegenError::Unsupported { op: "a computed index into registers" })
            }
        }
    }

    /// `position`'s parts as byte offsets into `buffer`, the placed one
    /// where the buffer's layout places it, and the immediate offset.
    fn byte_parts(
        &mut self,
        buffer: BufferId,
        position: Position,
    ) -> Result<(Vec<Ranked>, i64), CodegenError> {
        let dtype = self.kernel.buffers()[buffer].dtype;
        let size = i64::try_from(dtype.element_size().get()).map_err(|_| CodegenError::TooLarge)?;
        let constant = position.constant.checked_mul(size).ok_or(CodegenError::TooLarge)?;
        let mut parts = Vec::new();
        let immediate = match position.placed {
            Some(Ranked { rank, value }) => {
                let bytes = self.bytes(value, size);
                let (offset, immediate) = self.placed_offset(buffer, bytes, constant)?;
                parts.push(Ranked { rank, value: offset });
                immediate
            }
            None => self.swizzled_constant(buffer, constant)?,
        };
        for Ranked { rank, value } in position.shifted {
            let bytes = self.bytes(value, size);
            parts.push(Ranked { rank, value: bytes });
        }
        Ok((parts, immediate))
    }

    /// `elements` of `size` bytes each, in bytes.
    fn bytes(&mut self, elements: Reg<B64>, size: i64) -> Reg<B64> {
        let size = Value::<B64>::int(size);
        self.index_op(BinaryOp::Mul, Signedness::Signed, elements.into(), size)
    }

    /// Element `index` of `buffer`, as a held value.
    pub(super) fn load(&mut self, buffer: BufferId, index: ExprId) -> Result<Held, CodegenError> {
        let dtype = self.kernel.buffers()[buffer].dtype;
        let at = self.element(buffer, index)?;
        self.load_at(dtype, &at)
    }

    /// The element of `dtype` at `at`, as a held value.
    pub(super) fn load_at(&mut self, dtype: DType, at: &Place) -> Result<Held, CodegenError> {
        let ty = memory_type(dtype);
        Ok(match (dtype, Narrow::of(dtype)) {
            (DType::F32, _) => Held::Float(self.read::<F32>(at, ty)?),
            (_, Some(kind)) => {
                let bits = self.read::<B16>(at, ty)?;
                Held::Float(self.widened(bits, kind))
            }
            (DType::I64, _) => Held::Int64(self.read::<B64>(at, ty)?),
            (DType::Bool, _) => {
                let byte = self.read::<B32>(at, ty)?;
                let zero = Value::<B32>::int(0);
                Held::Bool(self.builder.compare_int(
                    Compare::Ne,
                    Signedness::Unsigned,
                    byte.into(),
                    zero,
                ))
            }
            _ => Held::Int32(self.read::<B32>(at, ty)?),
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
        let at = self.element(buffer, index)?;
        self.store_at(dtype, &at, value)
    }

    /// Stores `value` as the element of `dtype` at `at`.
    pub(super) fn store_at(
        &mut self,
        dtype: DType,
        at: &Place,
        value: Held,
    ) -> Result<(), CodegenError> {
        let ty = memory_type(dtype);
        let form = self.narrow_form(value, dtype);
        match form.unwrap_or_else(|| stored(&mut self.builder, value, dtype)) {
            Stored::F32(value) => self.write(at, ty, value),
            Stored::Half(value) => self.write(at, ty, value),
            Stored::Int32(value) => self.write(at, ty, value),
            Stored::Int64(value) => self.write(at, ty, value),
        }
    }

    /// `value`'s memory form if it was widened from `dtype`'s: storing it
    /// then needs no rounding, rounding a widened value being exact.
    fn narrow_form(&self, value: Held, dtype: DType) -> Option<Stored> {
        let Held::Float(wide) = value else { return None };
        let &(bits, kind) = self.narrow_forms.get(&wide)?;
        (Narrow::of(dtype) == Some(kind)).then_some(Stored::Half(bits))
    }
}
