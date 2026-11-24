//! `Stmt`: the lowered loop nest.

mod buffer;
mod fragment;
mod kernel;
mod swizzle;
mod value_expr;

pub use buffer::{Access, Buffer, BufferId, FragmentShape, MemScope};
pub use fragment::{FragmentTile, TileAccess, TileOrder};
pub use kernel::{KernelParam, KernelVar, LaunchShape, LoweredKernel};
pub use swizzle::Swizzle;
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
    /// Starts copying `run` adjacent elements of `source`, a global buffer,
    /// from `source_index` into `target`, a shared buffer, from
    /// `target_index`, as `cp.async` does: zeros where any of `bounds`
    /// fails. The elements land only once a wait completes the group the
    /// copy is committed in; until then they may not be read or written.
    AsyncCopy {
        /// The shared buffer written.
        target: BufferId,
        /// The first element written.
        target_index: ExprId,
        /// The global buffer read.
        source: BufferId,
        /// The first element read.
        source_index: ExprId,
        /// The adjacent elements copied.
        run: Dim,
        /// The conditions under which the elements are copied rather than
        /// zero.
        bounds: Vec<Condition>,
    },
    /// Closes the thread's group of asynchronous copies started since the
    /// last, as `cp.async.commit_group`.
    AsyncCommit,
    /// Waits until at most `pending` of the thread's committed groups are
    /// still in flight, as `cp.async.wait_group`; the copies of every
    /// other group have landed.
    AsyncWait {
        /// The groups that may stay in flight.
        pending: u32,
    },
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
    /// Whether `self` or any statement inside it is an `Mma`.
    pub(crate) fn has_mma(&self) -> bool {
        match self {
            Self::Seq(statements) => statements.iter().any(Self::has_mma),
            Self::For { body, .. }
            | Self::If { then: body, .. }
            | Self::Alloc { body, .. }
            | Self::LoadIndex { body, .. }
            | Self::FragmentEach { body, .. } => body.has_mma(),
            Self::Mma { .. } => true,
            Self::Store { .. }
            | Self::Barrier
            | Self::FragmentFill { .. }
            | Self::FragmentLoad { .. }
            | Self::AsyncCopy { .. }
            | Self::AsyncCommit
            | Self::AsyncWait { .. } => false,
        }
    }

    /// Whether `self` reads `buffer`, in a stored value or a gather index.
    pub(crate) fn reads(&self, buffer: BufferId) -> bool {
        match self {
            Self::Seq(statements) => statements.iter().any(|statement| statement.reads(buffer)),
            Self::For { body, .. } | Self::If { then: body, .. } | Self::Alloc { body, .. } => {
                body.reads(buffer)
            }
            Self::LoadIndex { buffer: read, body, .. } => *read == buffer || body.reads(buffer),
            Self::Store { value, .. } => value.reads(buffer),
            Self::Barrier
            | Self::FragmentFill { .. }
            | Self::AsyncCommit
            | Self::AsyncWait { .. } => false,
            Self::AsyncCopy { source, .. } | Self::FragmentLoad { source, .. } => *source == buffer,
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
    use crate::compute::{BinaryOp, Scalar};
    use crate::expr::{ExprId, VarId};
    use ptxc_core::{DType, Dim, Idx};
    use ptxc_target::{MmaInstruction, MmaShape};

    #[test]
    fn a_value_lists_every_load_in_it_with_its_index() {
        let (first, second) = (ExprId::new(1), ExprId::new(2));
        let load =
            |buffer, index| Box::new(ValueExpr::Load { buffer: BufferId::new(buffer), index });
        let sum = Box::new(ValueExpr::Binary(BinaryOp::Add, load(1, second), load(2, first)));
        let value =
            ValueExpr::Select { condition: load(0, first), then: sum, otherwise: load(3, first) };

        let loads = value.loads();

        let expected = [(0, first), (1, second), (2, first), (3, first)];
        assert_eq!(loads, expected.map(|(buffer, index)| (BufferId::new(buffer), index)));
    }

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
        let fill = Stmt::FragmentFill { fragment: whole(1), value: Scalar::bool(false) };
        let shape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(16) };
        let instruction =
            MmaInstruction { shape, multiplicands: DType::F16, accumulator: DType::F32 };
        let mma = Stmt::Mma { instruction, a: whole(3), b: whole(4), accumulator: whole(5) };

        assert_eq!([load.reads(buffer(2)), load.reads(buffer(1))], [true, false]);
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
    fn a_statement_has_an_mma_where_any_statement_inside_it_is_one() {
        let shape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(16) };
        let instruction =
            MmaInstruction { shape, multiplicands: DType::F16, accumulator: DType::F32 };
        let whole = |index| FragmentTile { buffer: BufferId::new(index), tile: ExprId::new(0) };
        let mma = Stmt::Mma { instruction, a: whole(0), b: whole(1), accumulator: whole(2) };
        let (var, index) = (VarId::new(0), ExprId::new(0));
        let around = |body: Stmt| {
            [
                Stmt::Seq(vec![Stmt::Barrier, body.clone()]),
                Stmt::For {
                    var,
                    extent: Dim::new(2),
                    kind: ForKind::Serial,
                    body: Box::new(body.clone()),
                },
                Stmt::If {
                    condition: Condition { index, bound: index },
                    then: Box::new(body.clone()),
                },
                Stmt::Alloc { buffer: BufferId::new(3), body: Box::new(body.clone()) },
                Stmt::LoadIndex {
                    var,
                    buffer: BufferId::new(4),
                    index,
                    body: Box::new(body.clone()),
                },
                Stmt::FragmentEach {
                    fragment: whole(2),
                    row: var,
                    column: var,
                    body: Box::new(body),
                },
            ]
        };
        let fill = Stmt::FragmentFill { fragment: whole(2), value: Scalar::bool(false) };

        assert!(around(mma).iter().all(Stmt::has_mma));
        assert!(!around(store_loading(1)).iter().any(Stmt::has_mma));
        assert!(!fill.has_mma() && !Stmt::Barrier.has_mma());
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
