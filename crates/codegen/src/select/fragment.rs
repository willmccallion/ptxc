//! Selects fragment statements element by element. Each lane keeps its
//! elements of a fragment's tiles in a local array, a tile's elements
//! together in the lane layout's order, which ptxas keeps in registers
//! where every tile index is a constant.

use super::lanes::{GROUP_LANES, LaneCoordinate, LaneLayout, Role};
use super::memory::{Place, Storage, place_in};
use super::{CodegenError, Selector};
use crate::ast::{
    Address, B64, BinaryOp, CvtMode, PtxType, Reg, Signedness, Space, Special, Value,
};
use ptxc_loop::{
    BufferId, Expr, FragmentTile, Interval, MemScope, Scalar, Stmt, TileAccess, TileOrder,
    ValueExpr, VarId,
};
use ptxc_target::{MmaInstruction, Target};
use std::collections::HashMap;

/// A fragment as the kernel's `mma`s use it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Fragment {
    pub(super) role: Role,
    pub(super) instruction: MmaInstruction,
}

impl Fragment {
    pub(super) const fn layout(self) -> LaneLayout {
        LaneLayout::of(self.role, self.instruction)
    }
}

/// The lane's index within its warp, its group (`laneid / 4`) and its
/// position in the group, as `s64`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Lane {
    pub(super) index: Reg<B64>,
    group: Reg<B64>,
    thread: Reg<B64>,
}

/// Every fragment the `mma`s in `body` use, by its role, each `mma` one
/// `target` supports.
pub(super) fn fragments(
    body: &Stmt,
    target: &Target,
) -> Result<HashMap<BufferId, Fragment>, CodegenError> {
    let mut mmas = Vec::new();
    collect_mmas(body, &mut mmas);
    let mut found = HashMap::new();
    for (instruction, operands) in mmas {
        let shape = instruction.shape;
        if target
            .mma_instruction(shape, instruction.multiplicands, instruction.accumulator)
            .is_none()
        {
            return Err(CodegenError::Unsupported { op: "an mma the target lacks" });
        }
        for (tile, role) in operands.into_iter().zip([Role::Lhs, Role::Rhs, Role::Accumulator]) {
            let fragment = Fragment { role, instruction };
            if found.insert(tile.buffer, fragment).is_some_and(|previous| previous != fragment) {
                return Err(CodegenError::Unsupported { op: "a fragment in two mma roles" });
            }
        }
    }
    Ok(found)
}

fn collect_mmas(stmt: &Stmt, mmas: &mut Vec<(MmaInstruction, [FragmentTile; 3])>) {
    match stmt {
        Stmt::Seq(stmts) => stmts.iter().for_each(|stmt| collect_mmas(stmt, mmas)),
        Stmt::For { body, .. }
        | Stmt::If { then: body, .. }
        | Stmt::Alloc { body, .. }
        | Stmt::LoadIndex { body, .. }
        | Stmt::FragmentEach { body, .. } => collect_mmas(body, mmas),
        Stmt::Mma { instruction, a, b, accumulator } => {
            mmas.push((*instruction, [*a, *b, *accumulator]));
        }
        Stmt::Store { .. }
        | Stmt::Barrier
        | Stmt::AsyncCopy { .. }
        | Stmt::AsyncCommit
        | Stmt::AsyncWait { .. }
        | Stmt::FragmentFill { .. }
        | Stmt::FragmentLoad { .. } => {}
    }
}

pub(super) fn signed(value: u64) -> Result<i64, CodegenError> {
    i64::try_from(value).map_err(|_| CodegenError::TooLarge)
}

impl Selector<'_> {
    /// Reads the lane's index, group and position, once at the start.
    pub(super) fn lane_indices(&mut self) -> Result<(), CodegenError> {
        let index = self.builder.special(Special::LaneId);
        let index: Reg<B64> =
            self.builder.convert(CvtMode::Plain, (PtxType::S64, PtxType::U32), index.into());
        let bits = Signedness::Bits;
        let shift = Value::<B64>::int(i64::from(GROUP_LANES.trailing_zeros()));
        let group = self.builder.int(BinaryOp::Shr, bits, index.into(), shift);
        let mask = Value::<B64>::int(signed(GROUP_LANES - 1)?);
        let thread = self.builder.int(BinaryOp::And, bits, index.into(), mask);
        self.lane = Some(Lane { index, group, thread });
        Ok(())
    }

    pub(super) fn lane(&self) -> Result<Lane, CodegenError> {
        self.lane.ok_or(CodegenError::Unbound)
    }

    pub(super) fn fragment(&self, buffer: BufferId) -> Result<Fragment, CodegenError> {
        let found = self.fragments.get(&buffer).copied();
        found.ok_or(CodegenError::Unsupported { op: "a fragment no mma uses" })
    }

    /// Where the lane's `element` of `fragment`'s tile is.
    pub(super) fn slot(
        &mut self,
        fragment: FragmentTile,
        element: u64,
    ) -> Result<Place, CodegenError> {
        let size = self.kernel.buffers()[fragment.buffer].dtype.element_size().get();
        let tile_bytes = signed(self.fragment(fragment.buffer)?.layout().elements() * size)?;
        let element_offset = signed(element * size)?;
        let storage = self.storage.get(&fragment.buffer).ok_or(CodegenError::Unbound)?;
        if let Expr::Const(tile) = self.kernel.exprs().get(fragment.tile) {
            let tile_offset = tile.checked_mul(tile_bytes).ok_or(CodegenError::TooLarge)?;
            let offset = tile_offset.checked_add(element_offset).ok_or(CodegenError::TooLarge)?;
            return place_in(storage, offset);
        }
        let Storage::Array { name, .. } = storage.clone() else {
            return Err(CodegenError::Unsupported { op: "a computed index into registers" });
        };
        let tile = self.index(fragment.tile)?;
        let signedness = Signedness::Signed;
        let scaled =
            self.builder.int(BinaryOp::Mul, signedness, tile.into(), Value::<B64>::int(tile_bytes));
        let base = self.builder.address_of(&name);
        let at = self.builder.int(BinaryOp::Add, signedness, base.into(), scaled.into());
        Ok(Place::Memory(Space::Local, Address::Reg { base: at.id(), offset: element_offset }))
    }

    /// `coordinate` for this lane.
    fn lane_coordinate(&mut self, coordinate: LaneCoordinate) -> Result<Reg<B64>, CodegenError> {
        let lane = self.lane()?;
        let offset = Value::<B64>::int(signed(coordinate.offset)?);
        let mut sum = self.builder.mov(PtxType::S64, offset);
        for (part, coefficient) in
            [(lane.group, coordinate.group), (lane.thread, coordinate.thread)]
        {
            if coefficient == 0 {
                continue;
            }
            let factor = Value::<B64>::int(signed(coefficient)?);
            let term = self.builder.int(BinaryOp::Mul, Signedness::Signed, part.into(), factor);
            sum = self.builder.int(BinaryOp::Add, Signedness::Signed, sum.into(), term.into());
        }
        Ok(sum)
    }

    /// The row and column of this lane's `element` under `layout`.
    fn lane_place(
        &mut self,
        layout: LaneLayout,
        element: u64,
    ) -> Result<[Reg<B64>; 2], CodegenError> {
        let [row, column] = layout.place(element);
        Ok([self.lane_coordinate(row)?, self.lane_coordinate(column)?])
    }

    /// The address in `source` of the tile's element at `row` and
    /// `column`, where `at` places the tile.
    pub(super) fn tile_address(
        &mut self,
        source: BufferId,
        at: TileAccess,
        place: [Reg<B64>; 2],
    ) -> Result<(Space, Address), CodegenError> {
        let (origin, constant) = self.swizzle_split(source, at.origin);
        let position = self.tile_position(TileAccess { origin, ..at }, place)?;
        self.address_at(source, (position, constant))
    }

    /// The position of the tile's element at `row` and `column` in the
    /// buffer `at` places the tile in.
    fn tile_position(
        &mut self,
        at: TileAccess,
        [row, column]: [Reg<B64>; 2],
    ) -> Result<Reg<B64>, CodegenError> {
        let (outer, inner) = match at.order {
            TileOrder::RowMajor => (row, column),
            TileOrder::Transposed => (column, row),
        };
        let stride = Value::<B64>::int(signed(at.stride.get())?);
        let origin = self.index_operand(at.origin)?;
        let signedness = Signedness::Signed;
        let scaled = self.builder.int(BinaryOp::Mul, signedness, outer.into(), stride);
        let within = self.builder.int(BinaryOp::Add, signedness, scaled.into(), inner.into());
        Ok(self.builder.int(BinaryOp::Add, signedness, within.into(), origin))
    }

    /// Sets each of the lane's elements of `fragment`'s tile to `value`.
    pub(super) fn fill_fragment(
        &mut self,
        fragment: FragmentTile,
        value: Scalar,
    ) -> Result<(), CodegenError> {
        let held = self.value(&ValueExpr::Const(value))?;
        let dtype = self.kernel.buffers()[fragment.buffer].dtype;
        for element in 0..self.fragment(fragment.buffer)?.layout().elements() {
            let slot = self.slot(fragment, element)?;
            self.store_at(dtype, &slot, held)?;
        }
        Ok(())
    }

    /// Loads the lane's elements of `fragment`'s tile one by one from
    /// `source`, where `at` places the tile.
    pub(super) fn load_elements(
        &mut self,
        fragment: FragmentTile,
        (source, at): (BufferId, TileAccess),
    ) -> Result<(), CodegenError> {
        let layout = self.fragment(fragment.buffer)?.layout();
        let dtype = self.kernel.buffers()[fragment.buffer].dtype;
        for element in 0..layout.elements() {
            let place = self.lane_place(layout, element)?;
            let (space, addr) = self.tile_address(source, at, place)?;
            let value =
                self.load_at(self.kernel.buffers()[source].dtype, &Place::Memory(space, addr))?;
            let slot = self.slot(fragment, element)?;
            self.store_at(dtype, &slot, value)?;
        }
        Ok(())
    }

    /// Runs `body` for each of the lane's elements of `fragment`'s tile,
    /// `row` and `column` bound to its place and `Element` to its value.
    pub(super) fn each_element(
        &mut self,
        fragment: FragmentTile,
        [row, column]: [VarId; 2],
        body: &Stmt,
    ) -> Result<(), CodegenError> {
        let layout = self.fragment(fragment.buffer)?.layout();
        let found = &self.kernel.buffers()[fragment.buffer];
        let dtype = found.dtype;
        if let MemScope::Fragment(shape) = found.scope {
            for (var, extent) in [(row, shape.rows), (column, shape.columns)] {
                if let Some(range) = Interval::of_loop(extent) {
                    self.ranges.insert(var, range);
                }
            }
        }
        for element in 0..layout.elements() {
            let [at_row, at_column] = self.lane_place(layout, element)?;
            let _row = self.vars.insert(row, at_row);
            let _column = self.vars.insert(column, at_column);
            let slot = self.slot(fragment, element)?;
            let value = self.load_at(dtype, &slot)?;
            let _previous = self.elements.insert(fragment.buffer, value);
            let outer = self.known.clone();
            self.statement(body)?;
            self.known.leave(outer);
        }
        let _last = self.elements.remove(&fragment.buffer);
        Ok(())
    }
}
