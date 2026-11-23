//! Exact evaluation of index expressions, the reference the simplifier is
//! tested against.

use super::{Expr, ExprArena, ExprId, VarId};

impl ExprArena {
    /// The value of `id` with each variable's value from `values`, or `None`
    /// if a variable has no value, an operation overflows `i64`, or a
    /// divisor is zero.
    pub fn evaluate(&self, id: ExprId, values: &impl Fn(VarId) -> Option<i64>) -> Option<i64> {
        let operands =
            |a: ExprId, b: ExprId| Some((self.evaluate(a, values)?, self.evaluate(b, values)?));
        match self.get(id) {
            Expr::Var(var) => values(var),
            Expr::Const(value) => Some(value),
            Expr::Add(a, b) => operands(a, b).and_then(|(a, b)| a.checked_add(b)),
            Expr::Mul(a, b) => operands(a, b).and_then(|(a, b)| a.checked_mul(b)),
            Expr::FloorDiv(a, b) => operands(a, b).and_then(|(a, b)| floor_div(a, b)),
            Expr::FloorMod(a, b) => operands(a, b).and_then(|(a, b)| floor_mod(a, b)),
            Expr::Min(a, b) => operands(a, b).map(|(a, b)| a.min(b)),
            Expr::Max(a, b) => operands(a, b).map(|(a, b)| a.max(b)),
        }
    }
}

/// `floor(a / b)`, or `None` if `b` is zero or the quotient overflows.
pub(super) fn floor_div(a: i64, b: i64) -> Option<i64> {
    let quotient = a.checked_div(b)?;
    let rounded_toward_zero = a % b != 0 && (a < 0) != (b < 0);
    if rounded_toward_zero { quotient.checked_sub(1) } else { Some(quotient) }
}

/// `a - b * floor(a / b)`, which has the sign of `b`, or `None` if `b` is
/// zero. It never overflows: `i64::MIN mod -1` is 0.
pub(crate) const fn floor_mod(a: i64, b: i64) -> Option<i64> {
    if b == 0 {
        return None;
    }
    let remainder = a.wrapping_rem(b);
    if remainder != 0 && (remainder < 0) != (b < 0) { Some(remainder + b) } else { Some(remainder) }
}
