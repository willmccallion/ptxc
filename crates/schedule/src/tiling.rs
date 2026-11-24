//! Tile shapes as newtypes.

use ptxc_core::ByteSize;
use ptxc_loop::StageCount;
use ptxc_target::MmaShape;

/// Rows by columns, of threads, of warps, or of the outputs one thread or
/// warp computes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct TileShape {
    pub(crate) rows: u64,
    pub(crate) columns: u64,
}

/// One block's share of a matrix product on CUDA cores: its threads, the
/// outputs each thread keeps in registers, how deep a slice of the
/// reduction it stages in shared memory per step, and the stages it
/// pipelines the steps over, if it does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct BlockTile {
    pub(crate) threads: TileShape,
    pub(crate) per_thread: TileShape,
    pub(crate) depth: u64,
    pub(crate) stages: Option<StageCount>,
}

/// How a block lays its staged tiles out in shared memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TileLayout {
    /// Row-major.
    Plain,
    /// Row-major with each tile's rows swizzled as CUTLASS's layouts are,
    /// so the rows an `ldmatrix` reads fall in different banks.
    Swizzled,
}

/// One block's share of a matrix product on tensor cores: its warps, the
/// `mma` tiles each warp keeps in fragments, how many `mma` steps deep a
/// slice of the reduction it stages in shared memory per step, the stages
/// it pipelines the steps over, if it does, and its tiles' layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct TensorCoreTile {
    pub(crate) warps: TileShape,
    pub(crate) per_warp: TileShape,
    pub(crate) steps: u64,
    pub(crate) stages: Option<StageCount>,
    pub(crate) layout: TileLayout,
}

/// The operand slices a block stages per step of a product's reduction,
/// `rows × depth` of the left operand and `depth × columns` of the right,
/// and the stages it pipelines the steps over, if it does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct StepTile {
    pub(crate) rows: u64,
    pub(crate) columns: u64,
    pub(crate) depth: u64,
    pub(crate) stages: Option<StageCount>,
}

impl StepTile {
    /// The bytes of both slices, of operands of `element` bytes each.
    pub(crate) const fn bytes(self, [lhs, rhs]: [ByteSize; 2]) -> ByteSize {
        ByteSize::new(self.depth * (self.rows * lhs.get() + self.columns * rhs.get()))
    }
}

impl BlockTile {
    /// The slices the block stages per step.
    pub(crate) const fn step(self) -> StepTile {
        StepTile {
            rows: self.threads.rows * self.per_thread.rows,
            columns: self.threads.columns * self.per_thread.columns,
            depth: self.depth,
            stages: self.stages,
        }
    }
}

impl TensorCoreTile {
    /// The slices the block stages per step, for `mma` tiles of `shape`.
    pub(crate) const fn step(self, shape: MmaShape) -> StepTile {
        StepTile {
            rows: self.warps.rows * self.per_warp.rows * shape.m.get(),
            columns: self.warps.columns * self.per_warp.columns * shape.n.get(),
            depth: self.steps * shape.k.get(),
            stages: self.stages,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockTile, StepTile, TensorCoreTile, TileLayout, TileShape};
    use ptxc_core::{ByteSize, Dim};
    use ptxc_target::MmaShape;

    const fn shape(rows: u64, columns: u64) -> TileShape {
        TileShape { rows, columns }
    }

    #[test]
    fn a_block_stages_its_threads_tiles_and_its_warps_mma_tiles_per_step() {
        let simt =
            BlockTile { threads: shape(16, 8), per_thread: shape(4, 2), depth: 8, stages: None };
        let warps = TensorCoreTile {
            warps: shape(2, 1),
            per_warp: shape(4, 8),
            steps: 3,
            stages: None,
            layout: TileLayout::Plain,
        };
        let mma = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(32) };

        let slices = [simt.step(), warps.step(mma)];

        let expected = [
            StepTile { rows: 64, columns: 16, depth: 8, stages: None },
            StepTile { rows: 128, columns: 64, depth: 96, stages: None },
        ];
        assert_eq!(slices, expected);
    }

    #[test]
    fn a_steps_bytes_count_each_slice_in_its_operands_element_size() {
        let slices = StepTile { rows: 64, columns: 16, depth: 8, stages: None };

        let bytes = slices.bytes([ByteSize::new(2), ByteSize::new(4)]);

        assert_eq!(bytes, ByteSize::new(8 * (64 * 2 + 16 * 4)));
    }
}
