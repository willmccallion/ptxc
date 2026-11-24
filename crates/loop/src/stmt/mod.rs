//! `Stmt`: the lowered loop nest.

mod buffer;
mod kernel;
mod value_expr;

pub use buffer::{Access, Buffer, BufferId, MemScope};
pub use kernel::{KernelParam, KernelVar, LaunchShape, LoweredKernel};
pub use value_expr::ValueExpr;
pub(crate) use value_expr::is_predicate;

use crate::expr::{ExprId, VarId};
use ptxc_core::Dim;

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
            Self::Barrier => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferId, Condition, ForKind, Stmt, ValueExpr};
    use crate::expr::{ExprId, VarId};
    use ptxc_core::{Dim, Idx};

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
