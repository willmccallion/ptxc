//! Integer index expressions: hash-consed in an `ExprArena`, evaluated
//! exactly, bounded by interval arithmetic, and simplified with the
//! variables' ranges (D4).

mod arena;
mod eval;

pub(crate) use eval::floor_mod;
mod interval;
mod linear;
mod simplify;

pub use arena::ExprArena;
pub use interval::{Interval, VarRanges};
pub(crate) use linear::LinearSum;
pub use simplify::Simplifier;

ptxc_core::define_index!(
    /// An expression in an `ExprArena`. Equal expressions have equal ids.
    pub ExprId
);

ptxc_core::define_index!(
    /// A variable an index expression reads: a loop variable, or an index
    /// loaded from memory.
    pub VarId
);

/// An integer index expression over `i64`. Division and modulo round
/// toward negative infinity, as TVM's `floordiv` and `floormod` do, so
/// `floordiv(x, c) * c + floormod(x, c) == x` for every `x`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Expr {
    /// A variable.
    Var(VarId),
    /// A constant.
    Const(i64),
    /// `a + b`.
    Add(ExprId, ExprId),
    /// `a * b`.
    Mul(ExprId, ExprId),
    /// `floor(a / b)`.
    FloorDiv(ExprId, ExprId),
    /// `a - b * floor(a / b)`, with the sign of `b`.
    FloorMod(ExprId, ExprId),
    /// The smaller of `a` and `b`.
    Min(ExprId, ExprId),
    /// The larger of `a` and `b`.
    Max(ExprId, ExprId),
}
