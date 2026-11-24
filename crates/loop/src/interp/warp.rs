//! The fragment statements: each runs once per warp of the active threads,
//! every lane of which must be active and agree on where the tile is.

use super::RunError;
use super::grid::Block;
use super::value::{Value, round, wrap};
use crate::compute::Scalar;
use crate::stmt::{BufferId, FragmentShape, MemScope, TileAccess};
use crate::verify::VerifyError;
use ptxc_core::DType;
use ptxc_core::float::Overflow;
use ptxc_target::MmaInstruction;
use std::collections::BTreeMap;

/// Threads in a warp, numbered consecutively from the block's x fastest.
const WARP: usize = 32;

/// The warp of the block's thread `thread`.
pub(super) const fn warp_of(thread: usize) -> usize {
    thread / WARP
}

/// A copy between a fragment and another buffer.
#[derive(Clone, Copy)]
pub(super) enum Tile {
    /// From `other` into `fragment`.
    Load { fragment: BufferId, other: BufferId, at: TileAccess },
    /// From `fragment` into `other`.
    Store { fragment: BufferId, other: BufferId, at: TileAccess },
}

impl Block<'_, '_> {
    /// Runs `run` once per warp of `active`, given the warp's first thread.
    fn each_warp(
        &mut self,
        active: &[usize],
        mut run: impl FnMut(&mut Self, usize) -> Result<(), RunError>,
    ) -> Result<(), RunError> {
        let mut lanes: BTreeMap<usize, usize> = BTreeMap::new();
        for &thread in active {
            *lanes.entry(warp_of(thread)).or_default() += 1;
        }
        for (warp, count) in lanes {
            if count != WARP {
                return Err(RunError::PartialWarp);
            }
            run(self, warp * WARP)?;
        }
        Ok(())
    }

    /// Sets every element of each warp's `fragment` to `value`.
    pub(super) fn fill(
        &mut self,
        fragment: BufferId,
        value: Scalar,
        active: &[usize],
    ) -> Result<(), RunError> {
        self.each_warp(active, |block, lead| block.fill_warp(fragment, value, lead))
    }

    fn fill_warp(
        &mut self,
        fragment: BufferId,
        value: Scalar,
        lead: usize,
    ) -> Result<(), RunError> {
        let shape = self.shape(fragment)?;
        for row in 0..shape.rows.get() {
            for column in 0..shape.columns.get() {
                self.store(fragment, position(shape, row, column)?, Value::of_scalar(value), lead)?;
            }
        }
        Ok(())
    }

    /// Copies `tile` element by element for each warp.
    pub(super) fn copy_tile(&mut self, tile: Tile, active: &[usize]) -> Result<(), RunError> {
        self.each_warp(active, |block, lead| block.copy_warp_tile(tile, lead))
    }

    fn copy_warp_tile(&mut self, tile: Tile, lead: usize) -> Result<(), RunError> {
        let (Tile::Load { fragment, other, at } | Tile::Store { fragment, other, at }) = tile;
        let shape = self.shape(fragment)?;
        let origin = self.warp_origin(at, lead)?;
        for row in 0..shape.rows.get() {
            for column in 0..shape.columns.get() {
                let elsewhere = at.position(origin, row, column).ok_or(RunError::Index)?;
                let own = position(shape, row, column)?;
                match tile {
                    Tile::Load { .. } => {
                        let value = self.load(other, elsewhere, lead)?;
                        self.store(fragment, own, value, lead)?;
                    }
                    Tile::Store { .. } => {
                        let value = self.load(fragment, own, lead)?;
                        self.store(other, elsewhere, value, lead)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// `accumulator += lhs · rhs` for each warp: the exact products added
    /// to the accumulator in `k` order, rounded once for a float
    /// accumulator and wrapped for an integer one.
    pub(super) fn mma(
        &mut self,
        instruction: MmaInstruction,
        operands: [BufferId; 3],
        active: &[usize],
    ) -> Result<(), RunError> {
        self.each_warp(active, |block, lead| block.mma_warp(instruction, operands, lead))
    }

    fn mma_warp(
        &mut self,
        instruction: MmaInstruction,
        [lhs, rhs, accumulator]: [BufferId; 3],
        lead: usize,
    ) -> Result<(), RunError> {
        let shape = instruction.shape;
        let [rows, columns, depth] = [shape.m, shape.n, shape.k].map(ptxc_core::Dim::get);
        let read = |block: &mut Self, buffer, row: u64, column: u64, columns: u64| {
            let position = i64::try_from(row * columns + column).map_err(|_| RunError::Index)?;
            block.load(buffer, position, lead)
        };
        for row in 0..rows {
            for column in 0..columns {
                let start = read(self, accumulator, row, column, columns)?;
                let mut sum = Sum::of(start, instruction.accumulator);
                for step in 0..depth {
                    let factor = read(self, lhs, row, step, depth)?;
                    sum.add(factor, read(self, rhs, step, column, columns)?);
                }
                let position =
                    i64::try_from(row * columns + column).map_err(|_| RunError::Index)?;
                self.store(accumulator, position, sum.result(instruction.accumulator), lead)?;
            }
        }
        Ok(())
    }

    /// Where the warp led by `lead` puts the tile, which every lane agrees on.
    fn warp_origin(&self, at: TileAccess, lead: usize) -> Result<i64, RunError> {
        let origin = self.index(at.origin, lead)?;
        for lane in lead..lead + WARP {
            if self.index(at.origin, lane)? != origin {
                return Err(RunError::WarpDisagrees);
            }
        }
        Ok(origin)
    }

    /// The shape of `fragment`'s tile.
    fn shape(&self, fragment: BufferId) -> Result<FragmentShape, RunError> {
        let found = &self.grid.kernel.buffers()[fragment];
        match found.scope {
            MemScope::Fragment(shape) => Ok(shape),
            _ => Err(RunError::Invalid(VerifyError::NotFragment(found.name.clone()))),
        }
    }
}

/// A running sum of products: exact for floats in `f64`, whose every
/// multiplicand type multiplies exactly, and for integers in `i64`.
enum Sum {
    Float(f64),
    Int(i64),
}

impl Sum {
    /// A sum starting at `start`, an accumulator of `dtype`.
    const fn of(start: Value, dtype: DType) -> Self {
        if dtype.is_float() { Self::Float(start.float()) } else { Self::Int(start.int()) }
    }

    const fn add(&mut self, lhs: Value, rhs: Value) {
        match self {
            Self::Float(sum) => *sum = lhs.float().mul_add(rhs.float(), *sum),
            Self::Int(sum) => *sum = sum.wrapping_add(lhs.int().wrapping_mul(rhs.int())),
        }
    }

    fn result(self, dtype: DType) -> Value {
        match self {
            Self::Float(sum) => round(sum, dtype, Overflow::NonFinite),
            Self::Int(sum) => wrap(i128::from(sum), dtype),
        }
    }
}

/// The position of row `row`, column `column` of a fragment of `shape`.
fn position(shape: FragmentShape, row: u64, column: u64) -> Result<i64, RunError> {
    row.checked_mul(shape.columns.get())
        .and_then(|offset| offset.checked_add(column))
        .and_then(|offset| i64::try_from(offset).ok())
        .ok_or(RunError::Index)
}
