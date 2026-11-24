//! Where fragment statements find their tiles: one tile of a fragment
//! buffer, and a tile's place in another buffer.

use super::BufferId;
use crate::expr::ExprId;
use ptxc_core::Dim;

/// One tile of a fragment buffer, which holds as many tiles as its size
/// has room for. Code generation needs `tile` to be a constant, as
/// unrolling the loops it reads makes it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FragmentTile {
    /// The fragment buffer.
    pub buffer: BufferId,
    /// Which of its tiles, the same in every lane of the warp.
    pub tile: ExprId,
}

/// Where a fragment's tile sits in a buffer: row `r`, column `c` at
/// `origin + r * stride + c`, or at `origin + c * stride + r` when the
/// order is `Transposed`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TileAccess {
    /// The first element's position, the same in every lane of the warp.
    pub origin: ExprId,
    /// The distance between rows as stored.
    pub stride: Dim,
    /// How the tile is laid out.
    pub order: TileOrder,
}

/// How a tile is laid out where a fragment loads or stores it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TileOrder {
    /// Row by row.
    RowMajor,
    /// Column by column: the tile's transpose, row by row.
    Transposed,
}

impl TileAccess {
    /// The position of row `row`, column `column` of the tile.
    pub(crate) fn position(self, origin: i64, row: u64, column: u64) -> Option<i64> {
        let (outer, inner) = match self.order {
            TileOrder::RowMajor => (row, column),
            TileOrder::Transposed => (column, row),
        };
        let offset = outer.checked_mul(self.stride.get())?.checked_add(inner)?;
        origin.checked_add(i64::try_from(offset).ok()?)
    }
}
