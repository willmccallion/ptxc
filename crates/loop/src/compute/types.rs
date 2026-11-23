//! The element types each scalar operation takes and gives. FP8 is
//! storage only, as in the graph IR: it reaches arithmetic through `Cast`.

use super::error::ComputeError;
use super::scalar_op::{BinaryOp, ReduceOp, UnaryOp};
use ptxc_core::DType;

const fn is_float(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::Bf16)
}

const fn is_integer(dtype: DType) -> bool {
    matches!(dtype, DType::I8 | DType::U8 | DType::I32 | DType::I64)
}

const fn is_numeric(dtype: DType) -> bool {
    is_float(dtype) || is_integer(dtype)
}

/// `Ok(result)` if `accepted`, otherwise `op` rejects `dtype`.
const fn accept(
    accepted: bool,
    op: &'static str,
    dtype: DType,
    result: DType,
) -> Result<DType, ComputeError> {
    if accepted { Ok(result) } else { Err(ComputeError::UnsupportedDType { op, dtype }) }
}

/// The type `op` gives for an operand of `dtype`.
pub(crate) const fn unary(op: UnaryOp, dtype: DType) -> Result<DType, ComputeError> {
    match op {
        UnaryOp::Abs | UnaryOp::Neg | UnaryOp::Sign => {
            accept(is_numeric(dtype), op.name(), dtype, dtype)
        }
        UnaryOp::Exp
        | UnaryOp::Expm1
        | UnaryOp::Log
        | UnaryOp::Log1p
        | UnaryOp::Sqrt
        | UnaryOp::Reciprocal
        | UnaryOp::Floor
        | UnaryOp::Ceil
        | UnaryOp::Round
        | UnaryOp::Erf
        | UnaryOp::Sin
        | UnaryOp::Cos
        | UnaryOp::Tanh => accept(is_float(dtype), op.name(), dtype, dtype),
        UnaryOp::Not => accept(matches!(dtype, DType::Bool), op.name(), dtype, DType::Bool),
        UnaryOp::IsNaN | UnaryOp::IsInf(_) => {
            accept(is_float(dtype), op.name(), dtype, DType::Bool)
        }
    }
}

/// The type `op` gives for operands of `lhs` and `rhs`, which must match.
pub(crate) fn binary(op: BinaryOp, lhs: DType, rhs: DType) -> Result<DType, ComputeError> {
    if lhs != rhs {
        return Err(ComputeError::MixedDTypes { op: op.name(), lhs, rhs });
    }
    let name = op.name();
    match op {
        BinaryOp::Add
        | BinaryOp::Sub
        | BinaryOp::Mul
        | BinaryOp::Div
        | BinaryOp::Max
        | BinaryOp::Min
        | BinaryOp::TruncatedRem
        | BinaryOp::Pow => accept(is_numeric(lhs), name, lhs, lhs),
        BinaryOp::FlooredRem => accept(is_integer(lhs), name, lhs, lhs),
        BinaryOp::Equal => accept(is_numeric(lhs) || lhs == DType::Bool, name, lhs, DType::Bool),
        BinaryOp::Less | BinaryOp::LessOrEqual | BinaryOp::Greater | BinaryOp::GreaterOrEqual => {
            accept(is_numeric(lhs), name, lhs, DType::Bool)
        }
        BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => {
            accept(lhs == DType::Bool, name, lhs, DType::Bool)
        }
    }
}

/// The type a reduction of `dtype` values gives.
pub(super) fn reduce(op: ReduceOp, dtype: DType) -> Result<DType, ComputeError> {
    let accepted = match op {
        ReduceOp::Sum | ReduceOp::Prod => is_numeric(dtype),
        ReduceOp::Max | ReduceOp::Min => is_numeric(dtype) || dtype == DType::Bool,
    };
    accept(accepted, op.name(), dtype, dtype)
}

/// Whether an arg-reduction can compare `dtype` values.
pub(super) const fn arg_comparable(dtype: DType) -> bool {
    is_numeric(dtype)
}
