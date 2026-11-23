//! Unary, binary and variadic elementwise ops, `Where` and `Cast`, each one
//! stage reading its operands broadcast to the output.

use super::NodeContext;
use crate::builder::Operand;
use crate::compose;
use crate::error::LowerError;
use crate::precision::{convert, demote, promote};
use crate::stage::LoweredGroup;
use ptxc_core::DType;
use ptxc_core::convert::count_to_f32;
use ptxc_graph::{BinaryOp, CastAttrs, InfDetection, ModKind, UnaryOp, VariadicOp};
use ptxc_loop::{BinaryOp as Scalar2, InfSign, Scalar, TensorExpr, UnaryOp as Scalar1};

pub(super) fn unary(cx: &NodeContext<'_>, op: UnaryOp) -> Result<LoweredGroup, LowerError> {
    cx.elementwise(|operands| {
        let [x] = cx.exact(operands)?;
        Ok(unary_expr(op, x))
    })
}

pub(super) fn binary(cx: &NodeContext<'_>, op: BinaryOp) -> Result<LoweredGroup, LowerError> {
    cx.elementwise(|operands| {
        let [lhs, rhs] = cx.exact(operands)?;
        Ok(binary_expr(op, lhs, rhs))
    })
}

pub(super) fn variadic(cx: &NodeContext<'_>, op: VariadicOp) -> Result<LoweredGroup, LowerError> {
    cx.elementwise(|operands| variadic_expr(op, operands).ok_or(LowerError::Operands(cx.id)))
}

/// `Where(condition, x, y)` as a select, which needs no arithmetic type.
pub(super) fn select(cx: &NodeContext<'_>) -> Result<LoweredGroup, LowerError> {
    cx.elementwise(|operands| {
        let [condition, then, otherwise] = cx.exact(operands)?;
        Ok(compose::select(condition.expr.clone(), then.expr.clone(), otherwise.expr.clone()))
    })
}

pub(super) fn cast(cx: &NodeContext<'_>, attrs: CastAttrs) -> Result<LoweredGroup, LowerError> {
    cx.elementwise(|operands| {
        let [x] = cx.exact(operands)?;
        if x.dtype == attrs.to() {
            return Ok(x.expr.clone());
        }
        let value = Box::new(x.expr.clone());
        Ok(TensorExpr::Cast { to: attrs.to(), overflow: attrs.overflow(), value })
    })
}

fn unary_expr(op: UnaryOp, x: &Operand) -> TensorExpr {
    let wide = || promote(x.expr.clone(), x.dtype);
    let arithmetic = match op {
        UnaryOp::Not => return compose::unary(Scalar1::Not, x.expr.clone()),
        UnaryOp::IsNaN => return compose::unary(Scalar1::IsNaN, wide()),
        UnaryOp::IsInf(detection) => return is_inf(detection, wide()),
        UnaryOp::Abs => Scalar1::Abs,
        UnaryOp::Neg => Scalar1::Neg,
        UnaryOp::Exp => Scalar1::Exp,
        UnaryOp::Log => Scalar1::Log,
        UnaryOp::Sqrt => Scalar1::Sqrt,
        UnaryOp::Reciprocal => Scalar1::Reciprocal,
        UnaryOp::Floor => Scalar1::Floor,
        UnaryOp::Ceil => Scalar1::Ceil,
        UnaryOp::Round => Scalar1::Round,
        UnaryOp::Sign => Scalar1::Sign,
        UnaryOp::Erf => Scalar1::Erf,
        UnaryOp::Sin => Scalar1::Sin,
        UnaryOp::Cos => Scalar1::Cos,
    };
    demote(compose::unary(arithmetic, wide()), x.dtype)
}

fn is_inf(detection: InfDetection, x: TensorExpr) -> TensorExpr {
    let sign = match detection {
        InfDetection::Both => InfSign::Either,
        InfDetection::Positive => InfSign::Positive,
        InfDetection::Negative => InfSign::Negative,
        InfDetection::Neither => return TensorExpr::Const(Scalar::bool(false)),
    };
    compose::unary(Scalar1::IsInf(sign), x)
}

fn binary_expr(op: BinaryOp, lhs: &Operand, rhs: &Operand) -> TensorExpr {
    let arithmetic = match op {
        BinaryOp::Pow => return pow(lhs, rhs),
        BinaryOp::Equal => return compare(Scalar2::Equal, lhs, rhs),
        BinaryOp::Less => return compare(Scalar2::Less, lhs, rhs),
        BinaryOp::LessOrEqual => return compare(Scalar2::LessOrEqual, lhs, rhs),
        BinaryOp::Greater => return compare(Scalar2::Greater, lhs, rhs),
        BinaryOp::GreaterOrEqual => return compare(Scalar2::GreaterOrEqual, lhs, rhs),
        BinaryOp::And => return compare(Scalar2::And, lhs, rhs),
        BinaryOp::Or => return compare(Scalar2::Or, lhs, rhs),
        BinaryOp::Xor => return compare(Scalar2::Xor, lhs, rhs),
        BinaryOp::Add => Scalar2::Add,
        BinaryOp::Sub => Scalar2::Sub,
        BinaryOp::Mul => Scalar2::Mul,
        BinaryOp::Div => Scalar2::Div,
        BinaryOp::Mod(ModKind::Integer) => Scalar2::FlooredRem,
        BinaryOp::Mod(ModKind::Fmod) => Scalar2::TruncatedRem,
    };
    let (a, b) = (promote(lhs.expr.clone(), lhs.dtype), promote(rhs.expr.clone(), rhs.dtype));
    demote(compose::binary(arithmetic, a, b), lhs.dtype)
}

/// A comparison or logical op, exact in the operands' own type.
fn compare(op: Scalar2, lhs: &Operand, rhs: &Operand) -> TensorExpr {
    compose::binary(op, lhs.expr.clone(), rhs.expr.clone())
}

/// `Pow`, whose exponent may have another type: a float base raises to the
/// exponent as f32; an integer base raises to an integer exponent in i64,
/// wrapping, and to a float exponent as f32, truncated.
fn pow(base: &Operand, exponent: &Operand) -> TensorExpr {
    if base.dtype.is_float() {
        let b = promote(base.expr.clone(), base.dtype);
        let e = convert(exponent.expr.clone(), exponent.dtype, DType::F32);
        return demote(compose::binary(Scalar2::Pow, b, e), base.dtype);
    }
    if exponent.dtype.is_float() {
        // NOTE: ptxc-interp, as ONNX Runtime, raises an integer base to a
        // float exponent in f64; loop IR has no f64, so ptxc uses f32, exact
        // while the base and the result stay below 2^24.
        let b = convert(base.expr.clone(), base.dtype, DType::F32);
        let e = promote(exponent.expr.clone(), exponent.dtype);
        let truncated = convert(compose::binary(Scalar2::Pow, b, e), DType::F32, DType::I64);
        return convert(truncated, DType::I64, base.dtype);
    }
    let b = convert(base.expr.clone(), base.dtype, DType::I64);
    let e = convert(exponent.expr.clone(), exponent.dtype, DType::I64);
    convert(compose::binary(Scalar2::Pow, b, e), DType::I64, base.dtype)
}

/// `Max`, `Min`, `Sum` or `Mean` of every operand, folded left to right in
/// the arithmetic type; `None` with no operands.
fn variadic_expr(op: VariadicOp, operands: &[Operand]) -> Option<TensorExpr> {
    let dtype = operands.first()?.dtype;
    let combine = match op {
        VariadicOp::Max => Scalar2::Max,
        VariadicOp::Min => Scalar2::Min,
        VariadicOp::Sum | VariadicOp::Mean => Scalar2::Add,
    };
    let folded = operands
        .iter()
        .map(|operand| promote(operand.expr.clone(), operand.dtype))
        .reduce(|acc, next| compose::binary(combine, acc, next))?;
    let result = match op {
        VariadicOp::Mean => {
            let count = compose::f32(count_to_f32(operands.len()));
            compose::binary(Scalar2::Div, folded, count)
        }
        VariadicOp::Max | VariadicOp::Min | VariadicOp::Sum => folded,
    };
    Some(demote(result, dtype))
}
