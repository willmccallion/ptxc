//! The fragment statements: each runs once per warp of the active threads,
//! every lane of which must be active and agree on where the tile is.

use super::RunError;
use super::grid::Block;
use super::value::{Value, round, wrap};
use crate::compute::Scalar;
use crate::expr::{ExprId, VarId};
use crate::stmt::{BufferId, FragmentShape, FragmentTile, MemScope, Stmt, TileAccess};
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

/// One warp's tile of a fragment buffer: its shape and where it starts.
#[derive(Clone, Copy)]
struct Placed {
    buffer: BufferId,
    shape: FragmentShape,
    start: i64,
}

impl Placed {
    /// The position of row `row`, column `column` in the buffer.
    fn position(self, row: u64, column: u64) -> Result<i64, RunError> {
        row.checked_mul(self.shape.columns.get())
            .and_then(|offset| offset.checked_add(column))
            .and_then(|offset| i64::try_from(offset).ok())
            .and_then(|offset| self.start.checked_add(offset))
            .ok_or(RunError::Index)
    }

    /// Every position of the tile with its row and column, row by row.
    fn elements(self) -> impl Iterator<Item = (u64, u64)> {
        let columns = self.shape.columns.get();
        (0..self.shape.rows.get())
            .flat_map(move |row| (0..columns).map(move |column| (row, column)))
    }
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

    /// Sets every element of each warp's tile of `fragment` to `value`.
    pub(super) fn fill(
        &mut self,
        fragment: FragmentTile,
        value: Scalar,
        active: &[usize],
    ) -> Result<(), RunError> {
        self.each_warp(active, |block, lead| {
            let placed = block.placed(fragment, lead)?;
            for (row, column) in placed.elements() {
                block.store(
                    placed.buffer,
                    placed.position(row, column)?,
                    Value::of_scalar(value),
                    lead,
                )?;
            }
            Ok(())
        })
    }

    /// Loads `fragment`'s tile element by element from `source`, where
    /// `at` places it, for each warp.
    pub(super) fn load_tile(
        &mut self,
        fragment: FragmentTile,
        (source, at): (BufferId, TileAccess),
        active: &[usize],
    ) -> Result<(), RunError> {
        self.each_warp(active, |block, lead| block.load_warp_tile(fragment, (source, at), lead))
    }

    fn load_warp_tile(
        &mut self,
        fragment: FragmentTile,
        (source, at): (BufferId, TileAccess),
        lead: usize,
    ) -> Result<(), RunError> {
        let placed = self.placed(fragment, lead)?;
        let origin = self.uniform(at.origin, lead)?;
        for (row, column) in placed.elements() {
            let elsewhere = at.position(origin, row, column).ok_or(RunError::Index)?;
            let value = self.load(source, elsewhere, lead)?;
            self.store(placed.buffer, placed.position(row, column)?, value, lead)?;
        }
        Ok(())
    }

    /// `accumulator += lhs · rhs` for each warp: the exact products added
    /// to the accumulator in `k` order, rounded once for a float
    /// accumulator and wrapped for an integer one.
    pub(super) fn mma(
        &mut self,
        instruction: MmaInstruction,
        operands: [FragmentTile; 3],
        active: &[usize],
    ) -> Result<(), RunError> {
        self.each_warp(active, |block, lead| block.mma_warp(instruction, operands, lead))
    }

    fn mma_warp(
        &mut self,
        instruction: MmaInstruction,
        operands: [FragmentTile; 3],
        lead: usize,
    ) -> Result<(), RunError> {
        let [lhs, rhs, accumulator] = operands.map(|operand| self.placed(operand, lead));
        let (lhs, rhs, accumulator) = (lhs?, rhs?, accumulator?);
        for (row, column) in accumulator.elements() {
            let own = accumulator.position(row, column)?;
            let start = self.load(accumulator.buffer, own, lead)?;
            let mut sum = Sum::of(start, instruction.accumulator);
            for step in 0..instruction.shape.k.get() {
                let factor = self.load(lhs.buffer, lhs.position(row, step)?, lead)?;
                sum.add(factor, self.load(rhs.buffer, rhs.position(step, column)?, lead)?);
            }
            self.store(accumulator.buffer, own, sum.result(instruction.accumulator), lead)?;
        }
        Ok(())
    }

    /// Runs `body` for each element of each warp's tile of `fragment`, in
    /// the warp's first lane, with `row` and `column` bound and the element
    /// what `ValueExpr::Element` reads.
    pub(super) fn each_element(
        &mut self,
        fragment: FragmentTile,
        [row_var, column_var]: [VarId; 2],
        body: &Stmt,
        active: &[usize],
    ) -> Result<(), RunError> {
        self.each_warp(active, |block, lead| {
            let placed = block.placed(fragment, lead)?;
            for (row, column) in placed.elements() {
                let as_index = |value: u64| i64::try_from(value).map_err(|_| RunError::Index);
                block.set_var(lead, row_var, as_index(row)?);
                block.set_var(lead, column_var, as_index(column)?);
                block.elements.push((placed.buffer, placed.position(row, column)?));
                let ran = block.exec(body, &[lead]);
                let _done = block.elements.pop();
                ran?;
            }
            Ok(())
        })
    }

    /// The element of `fragment` the innermost `FragmentEach` over it is at.
    pub(super) fn element(&mut self, fragment: BufferId, thread: usize) -> Result<Value, RunError> {
        let found = self.elements.iter().rev().find(|(buffer, _)| *buffer == fragment);
        let Some(&(_, position)) = found else {
            let name = self.grid.kernel.buffers()[fragment].name.clone();
            return Err(RunError::Invalid(VerifyError::ElementOutsideEach(name)));
        };
        self.load(fragment, position, thread)
    }

    /// The warp led by `lead`'s tile of `fragment`.
    fn placed(&self, fragment: FragmentTile, lead: usize) -> Result<Placed, RunError> {
        let found = &self.grid.kernel.buffers()[fragment.buffer];
        let MemScope::Fragment(shape) = found.scope else {
            return Err(RunError::Invalid(VerifyError::NotFragment(found.name.clone())));
        };
        let tiles = shape.rows.get().saturating_mul(shape.columns.get());
        let tile = self.uniform(fragment.tile, lead)?;
        let start = i64::try_from(tiles).ok().and_then(|tiles| tiles.checked_mul(tile));
        Ok(Placed { buffer: fragment.buffer, shape, start: start.ok_or(RunError::Index)? })
    }

    /// `index`'s value in the warp led by `lead`, which every lane agrees on.
    fn uniform(&self, index: ExprId, lead: usize) -> Result<i64, RunError> {
        let value = self.index(index, lead)?;
        for lane in lead..lead + WARP {
            if self.index(index, lane)? != value {
                return Err(RunError::WarpDisagrees);
            }
        }
        Ok(value)
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
