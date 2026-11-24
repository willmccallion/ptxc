//! Checks the fragment statements: every lane of a warp reaches them, they
//! take fragments of the shapes and types their operation does, and tiles
//! stay inside the buffers they are copied to and from.

use super::{Scope, VerifyError};
use crate::compute::Scalar;
use crate::stmt::{Access, BufferId, FragmentShape, MemScope, TileAccess};
use ptxc_core::{DType, Dim};
use ptxc_target::{MmaInstruction, ThreadsPerBlock};

/// Threads in a warp.
const WARP: u64 = 32;

impl Scope<'_> {
    pub(super) fn fill(&self, fragment: BufferId, value: Scalar) -> Result<(), VerifyError> {
        self.whole_warps()?;
        let _shape = self.fragment(fragment)?;
        same_type("a fragment fill", self.kernel.buffers()[fragment].dtype, value.dtype())
    }

    pub(super) fn tile_load(
        &self,
        fragment: BufferId,
        source: BufferId,
        at: TileAccess,
    ) -> Result<(), VerifyError> {
        self.whole_warps()?;
        let shape = self.fragment(fragment)?;
        let found = &self.kernel.buffers()[source];
        if found.scope != MemScope::Shared {
            return Err(VerifyError::TileNotShared(found.name.clone()));
        }
        self.tile(source, at, shape)?;
        same_type("a fragment load", self.kernel.buffers()[fragment].dtype, found.dtype)
    }

    pub(super) fn tile_store(
        &self,
        fragment: BufferId,
        target: BufferId,
        at: TileAccess,
    ) -> Result<(), VerifyError> {
        self.whole_warps()?;
        let shape = self.fragment(fragment)?;
        let found = &self.kernel.buffers()[target];
        let input = self
            .kernel
            .params()
            .iter()
            .any(|param| param.buffer == target && param.access == Access::Read);
        if input {
            return Err(VerifyError::StoreToInput(found.name.clone()));
        }
        self.tile(target, at, shape)?;
        let place = format!("a fragment store to {}", found.name);
        same_type(&place, found.dtype, self.kernel.buffers()[fragment].dtype)
    }

    /// `accumulator += a · b`: each fragment of the shape and type
    /// `instruction` takes.
    pub(super) fn mma(
        &self,
        instruction: MmaInstruction,
        [a, b, accumulator]: [BufferId; 3],
    ) -> Result<(), VerifyError> {
        self.whole_warps()?;
        let shape = instruction.shape;
        let tile = |rows: Dim, columns: Dim| FragmentShape { rows, columns };
        let operands = [
            (a, tile(shape.m, shape.k), instruction.multiplicands, "an mma's first multiplicand"),
            (b, tile(shape.k, shape.n), instruction.multiplicands, "an mma's second multiplicand"),
            (accumulator, tile(shape.m, shape.n), instruction.accumulator, "an mma's accumulator"),
        ];
        for (fragment, expected, dtype, place) in operands {
            let actual = self.fragment(fragment)?;
            if actual != expected {
                let name = self.kernel.buffers()[fragment].name.clone();
                return Err(VerifyError::FragmentShape { fragment: name, expected, actual });
            }
            same_type(place, dtype, self.kernel.buffers()[fragment].dtype)?;
        }
        Ok(())
    }

    /// Every lane of a warp reaches the statement: no condition on the
    /// thread is around it, and the block's threads fill whole warps.
    fn whole_warps(&self) -> Result<(), VerifyError> {
        if self.divergent {
            return Err(VerifyError::DivergentWarp);
        }
        let threads: u64 = self
            .kernel
            .launch()
            .block
            .iter()
            .map(|&extent| u64::from(ThreadsPerBlock::get(extent)))
            .product();
        if threads.is_multiple_of(WARP) { Ok(()) } else { Err(VerifyError::PartialWarp) }
    }

    /// The shape of `fragment`, a fragment allocated here.
    fn fragment(&self, fragment: BufferId) -> Result<FragmentShape, VerifyError> {
        let found = &self.kernel.buffers()[fragment];
        let MemScope::Fragment(shape) = found.scope else {
            return Err(VerifyError::NotFragment(found.name.clone()));
        };
        if self.allocated.contains(&fragment) {
            Ok(shape)
        } else {
            Err(VerifyError::NotAllocated(found.name.clone()))
        }
    }

    /// A tile of `shape` at `at` in `buffer`, which is not a fragment: its
    /// origin's variables bound, and within the buffer if it is scratch.
    fn tile(
        &self,
        buffer: BufferId,
        at: TileAccess,
        shape: FragmentShape,
    ) -> Result<(), VerifyError> {
        self.index(at.origin)?;
        let found = &self.kernel.buffers()[buffer];
        match found.scope {
            MemScope::Global => return Ok(()),
            MemScope::Fragment(_) => return Err(VerifyError::FragmentAccess(found.name.clone())),
            MemScope::Shared | MemScope::Local => {}
        }
        if !self.allocated.contains(&buffer) {
            return Err(VerifyError::NotAllocated(found.name.clone()));
        }
        let out_of_bounds = || VerifyError::OutOfBounds(found.name.clone());
        let origin = self.facts.bounds(self.kernel.exprs(), at.origin).ok_or_else(out_of_bounds)?;
        let last_row = shape.rows.get().checked_sub(1).ok_or_else(out_of_bounds)?;
        let last_column = shape.columns.get().checked_sub(1).ok_or_else(out_of_bounds)?;
        let last = at.position(origin.max(), last_row, last_column).ok_or_else(out_of_bounds)?;
        let size = i64::try_from(found.size.get()).map_err(|_| out_of_bounds())?;
        if origin.min() >= 0 && last < size { Ok(()) } else { Err(out_of_bounds()) }
    }
}

/// `actual` where `place` takes `expected`.
fn same_type(place: &str, expected: DType, actual: DType) -> Result<(), VerifyError> {
    if actual == expected {
        Ok(())
    } else {
        Err(VerifyError::WrongType { place: place.to_owned(), expected, actual })
    }
}
