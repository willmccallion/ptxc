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
