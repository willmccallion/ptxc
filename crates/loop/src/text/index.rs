//! Index expressions as text: `%i * 4 + %j`, `floordiv(%x, 8)`.

use crate::expr::{Expr, ExprArena, ExprId, VarId};
use std::fmt::Write;

/// Writes `id`, naming each variable with `name`. Sums and products are
/// infix, with a product's sum operands in parentheses; the rest are calls.
pub(super) fn write_index(
    out: &mut String,
    exprs: &ExprArena,
    id: ExprId,
    name: &impl Fn(VarId) -> String,
) {
    let call = |out: &mut String, function: &str, a: ExprId, b: ExprId| {
        out.push_str(function);
        out.push('(');
        write_index(out, exprs, a, name);
        out.push_str(", ");
        write_index(out, exprs, b, name);
        out.push(')');
    };
    match exprs.get(id) {
        Expr::Var(var) => {
            out.push('%');
            out.push_str(&name(var));
        }
        Expr::Const(value) => {
            let _infallible = write!(out, "{value}");
        }
        Expr::Add(a, b) => {
            write_index(out, exprs, a, name);
            out.push_str(" + ");
            write_index(out, exprs, b, name);
        }
        Expr::Mul(a, b) => {
            write_factor(out, exprs, a, name);
            out.push_str(" * ");
            write_factor(out, exprs, b, name);
        }
        Expr::FloorDiv(a, b) => call(out, "floordiv", a, b),
        Expr::FloorMod(a, b) => call(out, "floormod", a, b),
        Expr::Min(a, b) => call(out, "min", a, b),
        Expr::Max(a, b) => call(out, "max", a, b),
    }
}

/// Writes a product's operand, in parentheses if it is a sum.
fn write_factor(out: &mut String, exprs: &ExprArena, id: ExprId, name: &impl Fn(VarId) -> String) {
    if matches!(exprs.get(id), Expr::Add(..)) {
        out.push('(');
        write_index(out, exprs, id, name);
        out.push(')');
    } else {
        write_index(out, exprs, id, name);
    }
}
