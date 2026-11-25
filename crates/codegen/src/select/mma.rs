//! Selects `ldmatrix` for a multiplicand's fragment load where the tile's
//! rows are aligned for it, and `mma.sync` for each `mma`.

use super::fragment::{Fragment, signed};
use super::lanes::{BLOCK_ROWS, LaneLayout, Role, TileAxis};
use super::memory::Storage;
use super::{CodegenError, Selector};
use crate::ast::{
    B32, B64, BinaryOp, Class, F32, MatrixOrder, MmaOp, PtxType, Reg, Signedness, Space, Value,
};
use ptxc_core::DType;
use ptxc_loop::{BufferId, FragmentTile, TileAccess, TileOrder};
use ptxc_target::MmaInstruction;

/// The bytes of one row of an `ldmatrix` matrix, to which each row's
/// address must be aligned.
const MATRIX_ROW_BYTES: u64 = 16;

/// The lanes giving the row addresses of one `ldmatrix` matrix.
const MATRIX_LANES: u64 = 8;

impl Selector<'_> {
    /// Loads `fragment`'s tile from `source`, where `at` places it.
    pub(super) fn load_fragment(
        &mut self,
        fragment: FragmentTile,
        (source, at): (BufferId, TileAccess),
    ) -> Result<(), CodegenError> {
        let found = self.fragment(fragment.buffer)?;
        match self.matrix_order(found, source, at) {
            Some(order) => self.load_matrices(fragment, found.layout(), (source, at), order),
            None => self.load_elements(fragment, (source, at)),
        }
    }

    /// How `ldmatrix` loads `fragment` from `source` at `at`, if it can: a
    /// multiplicand in shared memory with 16-byte aligned rows, its runs
    /// adjacent in memory or, for 16-bit elements, transposed.
    fn matrix_order(
        &self,
        fragment: Fragment,
        source: BufferId,
        at: TileAccess,
    ) -> Option<MatrixOrder> {
        let layout = fragment.layout();
        let size = self.kernel.buffers()[source].dtype.element_size().get();
        let runs_adjacent = matches!(
            (layout.group_axis, at.order),
            (TileAxis::Rows, TileOrder::RowMajor) | (TileAxis::Columns, TileOrder::Transposed)
        );
        let order = match (runs_adjacent, size) {
            (true, _) => MatrixOrder::Rows,
            (false, 2) => MatrixOrder::Columns,
            (false, _) => return None,
        };
        let shared =
            matches!(self.storage.get(&source), Some(Storage::Array { space: Space::Shared, .. }));
        let aligned = (at.stride.get() * size).is_multiple_of(MATRIX_ROW_BYTES)
            && self.twos(at.origin, None) + size.trailing_zeros()
                >= MATRIX_ROW_BYTES.trailing_zeros();
        let usable = fragment.role != Role::Accumulator
            && shared
            && aligned
            && layout.group_blocks.is_power_of_two()
            && layout.blocks().is_power_of_two();
        usable.then_some(order)
    }

    /// Loads `fragment`'s tile with one `ldmatrix`, a matrix per block.
    fn load_matrices(
        &mut self,
        fragment: FragmentTile,
        layout: LaneLayout,
        (source, at): (BufferId, TileAccess),
        order: MatrixOrder,
    ) -> Result<(), CodegenError> {
        let [along_group, along_run] = self.matrix_row(layout, order)?;
        let place = match layout.group_axis {
            TileAxis::Rows => [along_group, along_run],
            TileAxis::Columns => [along_run, along_group],
        };
        let (_, addr) = self.tile_address(source, at, place)?;
        let registers = self.builder.ld_matrix(order, layout.blocks(), addr);
        for (block, register) in (0..).zip(registers) {
            let slot = self.slot(fragment, block * layout.run)?;
            self.write(&slot, PtxType::B32, register)?;
        }
        Ok(())
    }

    /// Where the matrix row whose address this lane gives `ldmatrix`
    /// starts, along the group's axis and the runs'. Lane `l` gives row
    /// `l % 8` of block `(l / 8) mod blocks`, so lanes past the last block
    /// repeat the first ones' addresses. A matrix's rows lie along the
    /// group's axis, or, transposed, along the runs'.
    fn matrix_row(
        &mut self,
        layout: LaneLayout,
        order: MatrixOrder,
    ) -> Result<[Reg<B64>; 2], CodegenError> {
        let lane = self.lane()?.index;
        let row = self.low_bits(lane, MATRIX_LANES)?;
        let matrix = self.shifted_right(lane, MATRIX_LANES.trailing_zeros());
        let block = self.low_bits(matrix, layout.blocks())?;
        let group_block = self.low_bits(block, layout.group_blocks)?;
        let run_block = self.shifted_right(block, layout.group_blocks.trailing_zeros());
        let along_group = self.scaled(group_block, BLOCK_ROWS)?;
        let along_run = self.scaled(run_block, layout.run_block_extent())?;
        let signed = Signedness::Signed;
        Ok(match order {
            MatrixOrder::Rows => {
                [self.index_op(BinaryOp::Add, signed, along_group.into(), row.into()), along_run]
            }
            MatrixOrder::Columns => {
                [along_group, self.index_op(BinaryOp::Add, signed, along_run.into(), row.into())]
            }
        })
    }

    /// `value mod count`, for a non-negative `value` and a power-of-two
    /// `count`.
    fn low_bits(&mut self, value: Reg<B64>, count: u64) -> Result<Reg<B64>, CodegenError> {
        let mask = signed(count - 1)?;
        Ok(self.index_op(BinaryOp::And, Signedness::Bits, value.into(), Value::<B64>::int(mask)))
    }

    /// `value >> shift`, for a non-negative `value`.
    fn shifted_right(&mut self, value: Reg<B64>, shift: u32) -> Reg<B64> {
        let shift = Value::<B64>::int(i64::from(shift));
        self.index_op(BinaryOp::Shr, Signedness::Bits, value.into(), shift)
    }

    fn scaled(&mut self, value: Reg<B64>, factor: u64) -> Result<Reg<B64>, CodegenError> {
        let factor = signed(factor)?;
        let signed = Signedness::Signed;
        Ok(self.index_op(BinaryOp::Mul, signed, value.into(), Value::<B64>::int(factor)))
    }

    /// `accumulator += a · b` on the warp's fragments, by `instruction`.
    pub(super) fn mma(
        &mut self,
        instruction: MmaInstruction,
        [a, b, accumulator]: [FragmentTile; 3],
    ) -> Result<(), CodegenError> {
        let op = mma_op(instruction)?;
        let a = self.multiplicand_registers(a)?;
        let b = self.multiplicand_registers(b)?;
        match op.accumulator {
            PtxType::S32 => self.accumulate::<B32>(op, [&a, &b], accumulator),
            _ => self.accumulate::<F32>(op, [&a, &b], accumulator),
        }
    }

    /// The registers of the lane's elements of `fragment`, a multiplicand:
    /// one 32-bit register a block.
    fn multiplicand_registers(
        &mut self,
        fragment: FragmentTile,
    ) -> Result<Vec<Reg<B32>>, CodegenError> {
        let layout = self.fragment(fragment.buffer)?.layout();
        (0..layout.blocks())
            .map(|block| {
                let slot = self.slot(fragment, block * layout.run)?;
                self.read::<B32>(&slot, PtxType::B32)
            })
            .collect()
    }

    /// Loads the lane's accumulators, multiplies `multiplicands` into them
    /// by `op`, and stores them back.
    fn accumulate<C: Class>(
        &mut self,
        op: MmaOp,
        multiplicands: [&[Reg<B32>]; 2],
        accumulator: FragmentTile,
    ) -> Result<(), CodegenError> {
        let layout = self.fragment(accumulator.buffer)?.layout();
        let slots = (0..layout.elements())
            .map(|element| self.slot(accumulator, element))
            .collect::<Result<Vec<_>, _>>()?;
        let ty = op.accumulator;
        let sums: Vec<Reg<C>> =
            slots.iter().map(|slot| self.read(slot, ty)).collect::<Result<_, _>>()?;
        let results = self.builder.mma(op, multiplicands, &sums);
        for (slot, result) in slots.iter().zip(results) {
            self.write(slot, ty, result)?;
        }
        Ok(())
    }
}

/// The `mma.sync` variant computing `instruction`.
const fn mma_op(instruction: MmaInstruction) -> Result<MmaOp, CodegenError> {
    let unsupported = CodegenError::Unsupported { op: "an mma of these types" };
    let multiplicands = match instruction.multiplicands {
        DType::F16 => PtxType::F16,
        DType::Bf16 => PtxType::Bf16,
        DType::F8E4M3Fn => PtxType::E4m3,
        DType::F8E5M2 => PtxType::E5m2,
        DType::I8 => PtxType::S8,
        DType::U8 => PtxType::U8,
        DType::F32 | DType::I32 | DType::I64 | DType::Bool => return Err(unsupported),
    };
    let accumulator = match instruction.accumulator {
        DType::F32 => PtxType::F32,
        DType::I32 => PtxType::S32,
        _ => return Err(unsupported),
    };
    Ok(MmaOp { shape: instruction.shape, accumulator, multiplicands })
}
