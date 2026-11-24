//! Tile shapes as newtypes.

/// Rows by columns, of threads, of warps, or of the outputs one thread or
/// warp computes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct TileShape {
    pub(crate) rows: u64,
    pub(crate) columns: u64,
}

/// One block's share of a matrix product on CUDA cores: its threads, the
/// outputs each thread keeps in registers, and how deep a slice of the
/// reduction it stages in shared memory per step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct BlockTile {
    pub(crate) threads: TileShape,
    pub(crate) per_thread: TileShape,
    pub(crate) depth: u64,
}

/// One block's share of a matrix product on tensor cores: its warps, the
/// `mma` tiles each warp keeps in fragments, and how many `mma` steps deep
/// a slice of the reduction it stages in shared memory per step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct TensorCoreTile {
    pub(crate) warps: TileShape,
    pub(crate) per_warp: TileShape,
    pub(crate) steps: u64,
}
