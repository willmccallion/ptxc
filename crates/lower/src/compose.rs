//! Constructors for the tensor expressions ops compose.

use ptxc_loop::{BinaryOp, Scalar, TensorExpr, UnaryOp};

pub(crate) fn unary(op: UnaryOp, x: TensorExpr) -> TensorExpr {
    TensorExpr::Unary(op, Box::new(x))
}

pub(crate) fn binary(op: BinaryOp, lhs: TensorExpr, rhs: TensorExpr) -> TensorExpr {
    TensorExpr::Binary(op, Box::new(lhs), Box::new(rhs))
}

/// `then` where `condition` holds, else `otherwise`.
pub(crate) fn select(condition: TensorExpr, then: TensorExpr, otherwise: TensorExpr) -> TensorExpr {
    TensorExpr::Select {
        condition: Box::new(condition),
        then: Box::new(then),
        otherwise: Box::new(otherwise),
    }
}

/// An `f32` constant.
pub(crate) fn f32(value: f32) -> TensorExpr {
    TensorExpr::Const(Scalar::f32(value))
}
