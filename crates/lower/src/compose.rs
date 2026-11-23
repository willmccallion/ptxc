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

pub(crate) fn neg(x: TensorExpr) -> TensorExpr {
    unary(UnaryOp::Neg, x)
}

pub(crate) fn add(lhs: TensorExpr, rhs: TensorExpr) -> TensorExpr {
    binary(BinaryOp::Add, lhs, rhs)
}

pub(crate) fn mul(lhs: TensorExpr, rhs: TensorExpr) -> TensorExpr {
    binary(BinaryOp::Mul, lhs, rhs)
}

pub(crate) fn div(lhs: TensorExpr, rhs: TensorExpr) -> TensorExpr {
    binary(BinaryOp::Div, lhs, rhs)
}

/// The larger value, NaN if either is NaN.
pub(crate) fn max(lhs: TensorExpr, rhs: TensorExpr) -> TensorExpr {
    binary(BinaryOp::Max, lhs, rhs)
}

/// The smaller value, NaN if either is NaN.
pub(crate) fn min(lhs: TensorExpr, rhs: TensorExpr) -> TensorExpr {
    binary(BinaryOp::Min, lhs, rhs)
}

pub(crate) fn less(lhs: TensorExpr, rhs: TensorExpr) -> TensorExpr {
    binary(BinaryOp::Less, lhs, rhs)
}

pub(crate) fn greater(lhs: TensorExpr, rhs: TensorExpr) -> TensorExpr {
    binary(BinaryOp::Greater, lhs, rhs)
}

pub(crate) fn at_least(lhs: TensorExpr, rhs: TensorExpr) -> TensorExpr {
    binary(BinaryOp::GreaterOrEqual, lhs, rhs)
}
