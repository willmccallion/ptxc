//! How a warp's lanes hold an `mma` operand's tile, as the PTX ISA's
//! matrix fragments for `mma.m16n8k16` and `mma.m16n8k32` lay them out.
//! The loop IR's fragments are logical tiles; only codegen knows these
//! layouts (D58).
//!
//! Every such layout has one shape: lane `l`, in group `g = l / 4` and
//! position `t = l % 4` within it, holds a run of adjacent elements along
//! one axis of each block of the tile, at index `g` along the other axis.
//! A block is 8 long along the group's axis and four runs long along the
//! other. Its elements are numbered run by run, blocks taken along the
//! group's axis first, as the ISA numbers `a0, a1, ...`.

use ptxc_target::MmaInstruction;

/// What a fragment is to the `mma`s that use it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum Role {
    /// `A`, `m × k`.
    Lhs,
    /// `B`, `k × n`.
    Rhs,
    /// `C` and `D`, `m × n`.
    Accumulator,
}

/// One of a tile's two axes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum TileAxis {
    Rows,
    Columns,
}

/// A coordinate that depends on the lane: `group · g + thread · t +
/// offset`, for the lane's group `g` and position `t` within it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct LaneCoordinate {
    pub(super) group: u64,
    pub(super) thread: u64,
    pub(super) offset: u64,
}

/// The lanes of a warp, among which a fragment's elements are shared.
pub(super) const WARP_LANES: u64 = 32;

/// The rows of a block along the group's axis.
pub(super) const BLOCK_ROWS: u64 = 8;

/// The lanes of a group, which share a block's row.
pub(super) const GROUP_LANES: u64 = 4;

/// How the lanes hold one operand's tile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct LaneLayout {
    /// The axis a lane's group indexes.
    pub(super) group_axis: TileAxis,
    /// The adjacent elements a lane holds per block.
    pub(super) run: u64,
    /// Blocks along the group's axis.
    pub(super) group_blocks: u64,
    /// Blocks along the other axis.
    pub(super) run_blocks: u64,
}

impl LaneLayout {
    /// The layout of `role` in `instruction`. A multiplicand's run fills
    /// 32 bits; the accumulator's is two elements.
    pub(super) const fn of(role: Role, instruction: MmaInstruction) -> Self {
        let shape = instruction.shape;
        let multiplicand_run = 4 / instruction.multiplicands.element_size().get();
        let (group_axis, run, [group_extent, run_extent]) = match role {
            Role::Lhs => (TileAxis::Rows, multiplicand_run, [shape.m, shape.k]),
            Role::Rhs => (TileAxis::Columns, multiplicand_run, [shape.n, shape.k]),
            Role::Accumulator => (TileAxis::Rows, 2, [shape.m, shape.n]),
        };
        Self {
            group_axis,
            run,
            group_blocks: group_extent.get() / BLOCK_ROWS,
            run_blocks: run_extent.get() / (GROUP_LANES * run),
        }
    }

    /// The blocks of the tile, each one 32-bit register of a multiplicand.
    pub(super) const fn blocks(self) -> u64 {
        self.group_blocks * self.run_blocks
    }

    /// The elements each lane holds.
    pub(super) const fn elements(self) -> u64 {
        self.blocks() * self.run
    }

    /// A block's extent along the runs' axis: a run for each lane of a
    /// group.
    pub(super) const fn run_block_extent(self) -> u64 {
        GROUP_LANES * self.run
    }

    /// Where `block` starts: its offsets along the group's axis and the
    /// other.
    pub(super) const fn block_origin(self, block: u64) -> [u64; 2] {
        let along_group = block % self.group_blocks * BLOCK_ROWS;
        let along_run = block / self.group_blocks * self.run_block_extent();
        [along_group, along_run]
    }

    /// The row and column of a lane's `element`.
    pub(super) const fn place(self, element: u64) -> [LaneCoordinate; 2] {
        let [along_group, along_run] = self.block_origin(element / self.run);
        let group = LaneCoordinate { group: 1, thread: 0, offset: along_group };
        let run =
            LaneCoordinate { group: 0, thread: self.run, offset: along_run + element % self.run };
        match self.group_axis {
            TileAxis::Rows => [group, run],
            TileAxis::Columns => [run, group],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LaneLayout, Role};
    use ptxc_core::{DType, Dim};
    use ptxc_target::{MmaInstruction, MmaShape, SmVersion, preset};
    use std::collections::HashSet;

    fn instruction(k: u64, multiplicands: DType, accumulator: DType) -> MmaInstruction {
        let shape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(k) };
        preset(SmVersion::SM_89)
            .unwrap()
            .mma_instruction(shape, multiplicands, accumulator)
            .unwrap()
    }

    /// Lane `lane`'s `element` under `layout`, as a row and column.
    fn place_of(layout: LaneLayout, lane: i64, element: u64) -> [i64; 2] {
        let (group, thread) = (lane >> 2, lane % 4);
        let signed = |value: u64| i64::try_from(value).unwrap();
        layout
            .place(element)
            .map(|at| signed(at.group) * group + signed(at.thread) * thread + signed(at.offset))
    }

    /// Checks every lane's every element against `expected`, the ISA's
    /// formula for element `i` given `groupID` and `threadID_in_group`,
    /// and that the lanes hold each of the tile's elements once.
    fn matches_isa(layout: LaneLayout, expected: impl Fn(i64, i64, i64) -> [i64; 2]) {
        let mut seen = HashSet::new();
        for lane in 0..32 {
            for element in 0..layout.elements() {
                let i = i64::try_from(element).unwrap();
                let place = place_of(layout, lane, element);
                assert_eq!(place, expected(lane >> 2, lane % 4, i), "lane {lane}, element {i}");
                assert!(seen.insert(place), "{place:?} held twice");
            }
        }
        assert_eq!(seen.len(), 32 * usize::try_from(layout.elements()).unwrap());
    }

    #[test]
    fn half_precision_fragments_follow_the_isas_m16n8k16_layout() {
        for multiplicands in [DType::F16, DType::Bf16] {
            let mma = instruction(16, multiplicands, DType::F32);

            matches_isa(LaneLayout::of(Role::Lhs, mma), |group, thread, i| {
                let row = if i < 2 || (4..6).contains(&i) { group } else { group + 8 };
                let column = if i < 4 { thread * 2 + (i & 1) } else { thread * 2 + (i & 1) + 8 };
                [row, column]
            });
            matches_isa(LaneLayout::of(Role::Rhs, mma), |group, thread, i| {
                let row = if i < 2 { thread * 2 + (i & 1) } else { thread * 2 + (i & 1) + 8 };
                [row, group]
            });
            matches_isa(LaneLayout::of(Role::Accumulator, mma), accumulator_place);
        }
    }

    #[test]
    fn eight_bit_fragments_follow_the_isas_m16n8k32_layout() {
        let cases = [
            (DType::I8, DType::I32),
            (DType::U8, DType::I32),
            (DType::F8E4M3Fn, DType::F32),
            (DType::F8E5M2, DType::F32),
        ];
        for (multiplicands, accumulator) in cases {
            let mma = instruction(32, multiplicands, accumulator);

            matches_isa(LaneLayout::of(Role::Lhs, mma), |group, thread, i| {
                let row = if i < 4 || (8..12).contains(&i) { group } else { group + 8 };
                let column = if i < 8 { thread * 4 + (i & 3) } else { thread * 4 + (i & 3) + 16 };
                [row, column]
            });
            matches_isa(LaneLayout::of(Role::Rhs, mma), |group, thread, i| {
                let row = if i < 4 { thread * 4 + (i & 3) } else { thread * 4 + (i & 3) + 16 };
                [row, group]
            });
            matches_isa(LaneLayout::of(Role::Accumulator, mma), accumulator_place);
        }
    }

    /// The ISA's accumulator layout, which every shape here shares.
    fn accumulator_place(group: i64, thread: i64, i: i64) -> [i64; 2] {
        let row = if i < 2 { group } else { group + 8 };
        [row, thread * 2 + (i & 1)]
    }

    #[test]
    fn a_multiplicands_blocks_are_one_register_each() {
        let mma = instruction(16, DType::F16, DType::F32);

        let [lhs, rhs] = [Role::Lhs, Role::Rhs].map(|role| LaneLayout::of(role, mma));

        assert_eq!([lhs.blocks(), rhs.blocks()], [4, 2]);
        assert_eq!(
            [lhs.block_origin(1), lhs.block_origin(2), rhs.block_origin(1)],
            [[8, 0], [0, 8], [0, 8]]
        );
    }
}
