//! What index expressions are known to be: how many factors of two every
//! value has, and how they step with a loop variable. Vector and matrix
//! accesses need their addresses aligned and their elements adjacent.

use super::Selector;
use ptxc_loop::{Expr, ExprId, VarId};

/// The most factors of two `twos` counts, past any an index can hold.
const MAX_TWOS: u32 = 63;

impl Selector<'_> {
    /// The most factors of two every value of index `id` is sure to have,
    /// with `zero`, if given, at 0.
    pub(super) fn twos(&self, id: ExprId, zero: Option<VarId>) -> u32 {
        match self.kernel.exprs().get(id) {
            Expr::Const(value) => value.trailing_zeros().min(MAX_TWOS),
            Expr::Var(var) if Some(var) == zero => MAX_TWOS,
            Expr::Var(_) | Expr::FloorDiv(..) => 0,
            // `floormod(a, b) = a - b * floordiv(a, b)`.
            Expr::Add(a, b) | Expr::Min(a, b) | Expr::Max(a, b) | Expr::FloorMod(a, b) => {
                self.twos(a, zero).min(self.twos(b, zero))
            }
            Expr::Mul(a, b) => (self.twos(a, zero) + self.twos(b, zero)).min(MAX_TWOS),
        }
    }

    /// How much index `id` grows for each step of `var`, if `var` is a
    /// term of a sum of terms: `None` where it is under a product, as an
    /// index scaling it has, or a division or a minimum.
    pub(super) fn step(&self, id: ExprId, var: VarId) -> Option<i64> {
        if !self.reads(id, var) {
            return Some(0);
        }
        match self.kernel.exprs().get(id) {
            Expr::Var(_) => Some(1),
            Expr::Add(a, b) => self.step(a, var)?.checked_add(self.step(b, var)?),
            _ => None,
        }
    }

    /// Whether index `id` reads `var`.
    fn reads(&self, id: ExprId, var: VarId) -> bool {
        match self.kernel.exprs().get(id) {
            Expr::Var(read) => read == var,
            Expr::Const(_) => false,
            Expr::Add(a, b)
            | Expr::Mul(a, b)
            | Expr::FloorDiv(a, b)
            | Expr::FloorMod(a, b)
            | Expr::Min(a, b)
            | Expr::Max(a, b) => self.reads(a, var) || self.reads(b, var),
        }
    }
}
