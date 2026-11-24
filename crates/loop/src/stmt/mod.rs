//! `Stmt`: the lowered loop nest.

mod buffer;
mod fragment;
mod kernel;
mod value_expr;

pub use buffer::{Access, Buffer, BufferId, FragmentShape, MemScope};
pub use fragment::{FragmentTile, TileAccess, TileOrder};
pub use kernel::{KernelParam, KernelVar, LaunchShape, LoweredKernel};
pub use value_expr::ValueExpr;
pub(crate) use value_expr::is_predicate;

use crate::compute::Scalar;
use crate::expr::{ExprId, VarId};
use ptxc_core::Dim;
use ptxc_target::MmaInstruction;

/// How a `For` runs its iterations.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ForKind {
    /// One after another.
    Serial,
    /// Fully unrolled.
    Unrolled,
    /// As one vector operation.
    Vectorized,
}

/// `index < bound`, over index expressions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Condition {
    /// The left side.
    pub index: ExprId,
    /// The bound it must stay below.
    pub bound: ExprId,
}

/// A statement of a lowered kernel: an owned tree, as TVM's TIR (D4).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Stmt {
    /// Statements in order.
    Seq(Vec<Self>),
    /// `var` from 0 to `extent - 1`.
    For {
        /// The loop variable.
        var: VarId,
        /// The iterations; shapes are static (D3).
        extent: Dim,
        /// How the iterations run.
        kind: ForKind,
        /// The body.
        body: Box<Self>,
    },
    /// `then` where `condition` holds.
    If {
        /// The condition.
        condition: Condition,
        /// The statement it guards.
        then: Box<Self>,
    },
    /// `buffer`, a shared or local buffer, for the extent of `body`.
    Alloc {
        /// The buffer.
        buffer: BufferId,
        /// Where it is live.
        body: Box<Self>,
    },
    /// Binds `var` to the integer `buffer[index]` for `body`, as a gather
    /// reads its index.
    LoadIndex {
        /// The variable bound.
        var: VarId,
        /// The integer buffer read.
        buffer: BufferId,
        /// The element's flat position.
        index: ExprId,
        /// Where `var` is bound.
        body: Box<Self>,
    },
    /// `buffer[index] = value`.
    Store {
        /// The buffer written.
        buffer: BufferId,
        /// The element's flat position.
        index: ExprId,
        /// The value stored.
        value: ValueExpr,
    },
    /// Waits for every thread of the block, as `__syncthreads`.
    Barrier,
    /// Sets every element of `fragment` to `value`; the whole warp
    /// executes it at once.
    FragmentFill {
        /// The fragment's tile.
        fragment: FragmentTile,
        /// The value of every element.
        value: Scalar,
    },
    /// Loads `fragment`'s tile from `source`, a shared buffer, at `at`; the
    /// whole warp executes it at once.
    FragmentLoad {
        /// The fragment's tile loaded.
        fragment: FragmentTile,
        /// The shared buffer the tile is in.
        source: BufferId,
        /// Where the tile is in `source`.
        at: TileAccess,
    },
    /// `accumulator += a · b` on fragments, as `instruction` computes it;
    /// the whole warp executes it at once. The interpreter adds the exact
    /// products to the accumulator in `k` order and rounds once; hardware
    /// sums them its own way, so the two agree exactly where the sum is
    /// exact in any order (D58).
    Mma {
        /// The shape and types.
        instruction: MmaInstruction,
        /// The `m × k` multiplicand.
        a: FragmentTile,
        /// The `k × n` multiplicand.
        b: FragmentTile,
        /// The `m × n` accumulator, read and written.
        accumulator: FragmentTile,
    },
    /// Stores `fragment`'s tile to `target` at `at`; the whole warp
    /// executes it at once.
    FragmentStore {
        /// The fragment's tile stored.
        fragment: FragmentTile,
        /// The buffer the tile goes to.
        target: BufferId,
        /// Where the tile goes in `target`.
        at: TileAccess,
    },
    /// Runs `body` once for each element of `fragment`'s tile, with `row`
    /// and `column` its position and `ValueExpr::Element` its value. The
    /// warp's lanes share the elements as code generation chooses, so each
    /// element's `body` runs in one lane.
    FragmentEach {
        /// The fragment's tile.
        fragment: FragmentTile,
        /// The element's row.
        row: VarId,
        /// The element's column.
        column: VarId,
        /// What each element runs.
        body: Box<Self>,
    },
}

impl Stmt {
    /// Whether `self` reads `buffer`, in a stored value or a gather index.
    pub(crate) fn reads(&self, buffer: BufferId) -> bool {
        match self {
            Self::Seq(statements) => statements.iter().any(|statement| statement.reads(buffer)),
            Self::For { body, .. } | Self::If { then: body, .. } | Self::Alloc { body, .. } => {
                body.reads(buffer)
            }
            Self::LoadIndex { buffer: read, body, .. } => *read == buffer || body.reads(buffer),
            Self::Store { value, .. } => value.reads(buffer),
            Self::Barrier | Self::FragmentFill { .. } => false,
            Self::FragmentLoad { source, .. } => *source == buffer,
            Self::FragmentStore { fragment, .. } => fragment.buffer == buffer,
            Self::Mma { a, b, accumulator, .. } => {
                [a, b, accumulator].iter().any(|operand| operand.buffer == buffer)
            }
            Self::FragmentEach { fragment, body, .. } => {
                fragment.buffer == buffer || body.reads(buffer)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BufferId, Condition, ForKind, FragmentTile, Stmt, TileAccess, TileOrder, ValueExpr,
    };
    use crate::compute::Scalar;
    use crate::expr::{ExprId, VarId};
    use ptxc_core::{DType, Dim, Idx};
    use ptxc_target::{MmaInstruction, MmaShape};

    fn store_loading(buffer: usize) -> Stmt {
        let index = ExprId::new(0);
        let value = ValueExpr::Load { buffer: BufferId::new(buffer), index };
        Stmt::Store { buffer: BufferId::new(9), index, value }
    }

    fn gather(buffer: usize, body: Stmt) -> Stmt {
        let (var, index) = (VarId::new(0), ExprId::new(0));
        Stmt::LoadIndex { var, buffer: BufferId::new(buffer), index, body: Box::new(body) }
    }

    #[test]
    fn a_statement_reads_what_its_values_and_gathers_load() {
        let tile = BufferId::new(1);

        assert!(store_loading(1).reads(tile));
        assert!(!store_loading(2).reads(tile));
        assert!(!Stmt::Barrier.reads(tile));
        assert!(gather(1, Stmt::Barrier).reads(tile));
        assert!(gather(2, store_loading(1)).reads(tile));
        assert!(!gather(2, store_loading(2)).reads(tile));
    }

    #[test]
    fn a_fragment_statement_reads_the_buffers_it_copies_or_multiplies_from() {
        let at =
            TileAccess { origin: ExprId::new(0), stride: Dim::new(16), order: TileOrder::RowMajor };
        let buffer = BufferId::new;
        let whole = |index| FragmentTile { buffer: buffer(index), tile: ExprId::new(0) };
        let load = Stmt::FragmentLoad { fragment: whole(1), source: buffer(2), at };
        let store = Stmt::FragmentStore { fragment: whole(1), target: buffer(2), at };
        let fill = Stmt::FragmentFill { fragment: whole(1), value: Scalar::bool(false) };
        let shape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(16) };
        let instruction =
            MmaInstruction { shape, multiplicands: DType::F16, accumulator: DType::F32 };
        let mma = Stmt::Mma { instruction, a: whole(3), b: whole(4), accumulator: whole(5) };

        assert_eq!([load.reads(buffer(2)), load.reads(buffer(1))], [true, false]);
        assert_eq!([store.reads(buffer(1)), store.reads(buffer(2))], [true, false]);
        assert!(!fill.reads(buffer(1)));
        assert!((3..=5).all(|operand| mma.reads(buffer(operand))) && !mma.reads(buffer(6)));
        let each = |body| Stmt::FragmentEach {
            fragment: whole(1),
            row: VarId::new(0),
            column: VarId::new(1),
            body: Box::new(body),
        };
        assert!(each(Stmt::Barrier).reads(buffer(1)));
        assert!(each(store_loading(7)).reads(buffer(7)));
        assert!(!each(Stmt::Barrier).reads(buffer(7)));
    }

    #[test]
    fn a_statement_reads_what_any_statement_inside_it_reads() {
        let (tile, index) = (BufferId::new(1), ExprId::new(0));
        let inside = |body: Stmt| {
            [
                Stmt::Seq(vec![Stmt::Barrier, body.clone()]),
                Stmt::For {
                    var: VarId::new(0),
                    extent: Dim::new(2),
                    kind: ForKind::Serial,
                    body: Box::new(body.clone()),
                },
                Stmt::If {
                    condition: Condition { index, bound: index },
                    then: Box::new(body.clone()),
                },
                Stmt::Alloc { buffer: BufferId::new(3), body: Box::new(body) },
            ]
        };

        assert!(inside(store_loading(1)).iter().all(|outer| outer.reads(tile)));
        assert!(inside(store_loading(2)).iter().all(|outer| !outer.reads(tile)));
    }
}
