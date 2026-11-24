//! Checks the fragment statements: every lane of a warp reaches them, they
//! take fragments of the shapes and types their operation does, and tiles
//! stay inside the buffers they are copied to and from.

use super::{Scope, VerifyError, same_type};
use crate::compute::Scalar;
use crate::expr::{Interval, VarId};
use crate::stmt::{BufferId, FragmentShape, FragmentTile, MemScope, Stmt, TileAccess};
use ptxc_core::{DType, Dim};
use ptxc_target::{MmaInstruction, ThreadsPerBlock};

/// Threads in a warp.
const WARP: u64 = 32;

impl Scope<'_> {
    pub(super) fn fill(&self, fragment: FragmentTile, value: Scalar) -> Result<(), VerifyError> {
        self.whole_warps()?;
        let _shape = self.fragment(fragment)?;
        same_type("a fragment fill", self.dtype(fragment), value.dtype())
    }

    /// `body` for each element of `fragment`'s tile: `row` and `column`
    /// bound within the tile, the element readable, and no barrier or
    /// fragment statement inside, as each element runs in one lane.
    pub(super) fn each(
        &self,
        fragment: FragmentTile,
        [row, column]: [VarId; 2],
        body: &Stmt,
    ) -> Result<(), VerifyError> {
        self.whole_warps()?;
        let shape = self.fragment(fragment)?;
        let mut inner = self.clone();
        for (var, extent) in [(row, shape.rows), (column, shape.columns)] {
            let _new = inner.bound.insert(var);
            if let Some(range) = Interval::of_loop(extent) {
                inner.facts.insert_range(var, range);
            }
        }
        inner.divergent = true;
        inner.elements.push(fragment.buffer);
        inner.stmt(body)
    }

    /// The type of `fragment`'s element, read inside a `FragmentEach` over
    /// it.
    pub(super) fn element(&self, fragment: BufferId) -> Result<DType, VerifyError> {
        let found = &self.kernel.buffers()[fragment];
        if self.elements.contains(&fragment) {
            Ok(found.dtype)
        } else {
            Err(VerifyError::ElementOutsideEach(found.name.clone()))
        }
    }

    pub(super) fn tile_load(
        &self,
        fragment: FragmentTile,
        source: BufferId,
        at: TileAccess,
    ) -> Result<(), VerifyError> {
        self.whole_warps()?;
        let shape = self.fragment(fragment)?;
        let found = &self.kernel.buffers()[source];
        if !matches!(found.scope, MemScope::Shared { .. }) {
            return Err(VerifyError::TileNotShared(found.name.clone()));
        }
        self.tile(source, at, shape)?;
        same_type("a fragment load", self.dtype(fragment), found.dtype)
    }

    /// `accumulator += a · b`: each fragment of the shape and type
    /// `instruction` takes.
    pub(super) fn mma(
        &self,
        instruction: MmaInstruction,
        [a, b, accumulator]: [FragmentTile; 3],
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
                let name = self.kernel.buffers()[fragment.buffer].name.clone();
                return Err(VerifyError::FragmentShape { fragment: name, expected, actual });
            }
            same_type(place, dtype, self.dtype(fragment))?;
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

    /// The shape of `fragment`'s tiles, of a fragment allocated here, the
    /// tile one of those it holds.
    fn fragment(&self, fragment: FragmentTile) -> Result<FragmentShape, VerifyError> {
        let found = &self.kernel.buffers()[fragment.buffer];
        let MemScope::Fragment(shape) = found.scope else {
            return Err(VerifyError::NotFragment(found.name.clone()));
        };
        if !self.allocated.contains(&fragment.buffer) {
            return Err(VerifyError::NotAllocated(found.name.clone()));
        }
        self.index(fragment.tile)?;
        let elements = shape.rows.get().saturating_mul(shape.columns.get());
        let tiles =
            found.size.get().checked_div(elements).and_then(|tiles| i64::try_from(tiles).ok());
        let tile = self.facts.bounds(self.kernel.exprs(), fragment.tile);
        match tile.zip(tiles) {
            Some((tile, tiles)) if tile.min() >= 0 && tile.max() < tiles => Ok(shape),
            _ => Err(VerifyError::OutOfBounds(found.name.clone())),
        }
    }

    /// The element type of `fragment`'s buffer.
    fn dtype(&self, fragment: FragmentTile) -> DType {
        self.kernel.buffers()[fragment.buffer].dtype
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
            MemScope::Shared { .. } | MemScope::Local => {}
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
