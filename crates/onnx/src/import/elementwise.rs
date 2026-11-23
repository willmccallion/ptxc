//! Imports unary, binary and variadic elementwise ops, `Where`, and casts.

use crate::context::NodeContext;
use crate::error::ImportError;
use crate::lowered::Lowered;
use crate::tensor::dtype_from_onnx;
use ptxc_core::float::Overflow;
use ptxc_graph::{BinaryOp, CastAttrs, InfDetection, ModKind, Op, UnaryOp, VariadicOp};

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    if let Some(unary) = unary(op) {
        return Some(Ok(Lowered::on_inputs(Op::Unary(unary), 1)));
    }
    if let Some(binary) = binary(op) {
        return Some(Ok(Lowered::on_inputs(Op::Binary(binary), 2)));
    }
    if let Some(variadic) = variadic(op) {
        return Some(Ok(Lowered::on_inputs(Op::Variadic(variadic), context.input_count())));
    }
    Some(match op {
        "IsInf" => is_inf(context),
        "Mod" => context.attrs.int_or("fmod", 0).map(|fmod| {
            let kind = if fmod == 0 { ModKind::Integer } else { ModKind::Fmod };
            Lowered::on_inputs(Op::Binary(BinaryOp::Mod(kind)), 2)
        }),
        "Where" => Ok(Lowered::on_inputs(Op::Where, 3)),
        "Cast" => cast(context, CastTarget::Attribute),
        "CastLike" => cast(context, CastTarget::SecondInput),
        _ => return None,
    })
}

fn unary(op: &str) -> Option<UnaryOp> {
    Some(match op {
        "Abs" => UnaryOp::Abs,
        "Neg" => UnaryOp::Neg,
        "Exp" => UnaryOp::Exp,
        "Log" => UnaryOp::Log,
        "Sqrt" => UnaryOp::Sqrt,
        "Reciprocal" => UnaryOp::Reciprocal,
        "Floor" => UnaryOp::Floor,
        "Ceil" => UnaryOp::Ceil,
        "Round" => UnaryOp::Round,
        "Sign" => UnaryOp::Sign,
        "Erf" => UnaryOp::Erf,
        "Sin" => UnaryOp::Sin,
        "Cos" => UnaryOp::Cos,
        "Not" => UnaryOp::Not,
        "IsNaN" => UnaryOp::IsNaN,
        _ => return None,
    })
}

fn binary(op: &str) -> Option<BinaryOp> {
    Some(match op {
        "Add" => BinaryOp::Add,
        "Sub" => BinaryOp::Sub,
        "Mul" => BinaryOp::Mul,
        "Div" => BinaryOp::Div,
        "Pow" => BinaryOp::Pow,
        "Equal" => BinaryOp::Equal,
        "Less" => BinaryOp::Less,
        "LessOrEqual" => BinaryOp::LessOrEqual,
        "Greater" => BinaryOp::Greater,
        "GreaterOrEqual" => BinaryOp::GreaterOrEqual,
        "And" => BinaryOp::And,
        "Or" => BinaryOp::Or,
        "Xor" => BinaryOp::Xor,
        _ => return None,
    })
}

fn variadic(op: &str) -> Option<VariadicOp> {
    Some(match op {
        "Max" => VariadicOp::Max,
        "Min" => VariadicOp::Min,
        "Sum" => VariadicOp::Sum,
        "Mean" => VariadicOp::Mean,
        _ => return None,
    })
}

fn is_inf(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let positive = context.attrs.int_or("detect_positive", 1)? != 0;
    let negative = context.attrs.int_or("detect_negative", 1)? != 0;
    let detect = match (positive, negative) {
        (true, true) => InfDetection::Both,
        (true, false) => InfDetection::Positive,
        (false, true) => InfDetection::Negative,
        (false, false) => InfDetection::Neither,
    };
    Ok(Lowered::on_inputs(Op::Unary(UnaryOp::IsInf(detect)), 1))
}

/// Where a cast's target type comes from.
#[derive(Clone, Copy)]
enum CastTarget {
    /// `Cast`'s `to` attribute.
    Attribute,
    /// `CastLike`'s second input's element type.
    SecondInput,
}

/// `Cast` or `CastLike`. `saturate` affects only FP8 targets; `round_mode`
/// (opset 25) only an element type ptxc does not support, so any value but
/// the default is unsupported.
fn cast(context: &mut NodeContext<'_>, target: CastTarget) -> Result<Lowered, ImportError> {
    let to = match target {
        CastTarget::Attribute => {
            let code = context.attrs.int("to")?.ok_or_else(|| context.attrs.unsupported("to"))?;
            dtype_from_onnx(i32::try_from(code).map_err(|_| context.attrs.unsupported("to"))?)?
        }
        CastTarget::SecondInput => context.ty(1)?.dtype(),
    };
    let overflow = if context.attrs.int_or("saturate", 1)? == 0 {
        Overflow::NonFinite
    } else {
        Overflow::Saturate
    };
    if context.attrs.string_or("round_mode", "up")? != "up" {
        return Err(context.attrs.unsupported("round_mode"));
    }
    Ok(Lowered::on_inputs(Op::Cast(CastAttrs::new(to, overflow)), 1))
}
