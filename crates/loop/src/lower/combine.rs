//! How reductions combine: identities, and the pair rule arg-reductions
//! use, which gives the same answer in any order.

use crate::compute::{ArgKind, BinaryOp, ReduceOp, Scalar, TieBreak, UnaryOp};
use crate::stmt::ValueExpr;
use ptxc_core::DType;

/// The value a reduction starts from: 0 for sums, 1 for products, the
/// type's lowest value for `Max` and highest for `Min`.
pub(super) const fn identity(op: ReduceOp, dtype: DType) -> Scalar {
    match op {
        ReduceOp::Sum => Scalar::zero(dtype),
        ReduceOp::Prod => Scalar::one(dtype),
        ReduceOp::Max => Scalar::lowest(dtype),
        ReduceOp::Min => Scalar::highest(dtype),
    }
}

/// The value an arg-reduction's best starts from, which every value but
/// another extreme beats.
pub(super) const fn arg_start(kind: ArgKind, dtype: DType) -> Scalar {
    match kind {
        ArgKind::Max => Scalar::lowest(dtype),
        ArgKind::Min => Scalar::highest(dtype),
    }
}

/// The position an arg-reduction's best starts from, which every real
/// position beats in a tie.
pub(super) const fn position_start(tie: TieBreak) -> Scalar {
    match tie {
        TieBreak::First => Scalar::highest(DType::I64),
        TieBreak::Last => Scalar::lowest(DType::I64),
    }
}

/// `accumulated op value`, with `Max` and `Min` as `Or` and `And` on
/// booleans.
pub(super) fn combine(
    op: ReduceOp,
    dtype: DType,
    accumulated: ValueExpr,
    value: ValueExpr,
) -> ValueExpr {
    let binary = match (op, dtype) {
        (ReduceOp::Sum, _) => BinaryOp::Add,
        (ReduceOp::Prod, _) => BinaryOp::Mul,
        (ReduceOp::Max, DType::Bool) => BinaryOp::Or,
        (ReduceOp::Min, DType::Bool) => BinaryOp::And,
        (ReduceOp::Max, _) => BinaryOp::Max,
        (ReduceOp::Min, _) => BinaryOp::Min,
    };
    ValueExpr::Binary(binary, Box::new(accumulated), Box::new(value))
}

/// One candidate of an arg-reduction: a value and its position.
pub(super) struct Candidate {
    pub(super) value: ValueExpr,
    pub(super) position: ValueExpr,
}

/// Whether `challenger` replaces `best`: a NaN beats any number, and of
/// two NaNs the earlier position wins; otherwise the strictly better value
/// wins, and the tie-break decides between equal values by position. It
/// depends only on values and positions, so any order of comparisons, and
/// any split across threads, gives the same result.
pub(super) fn replaces(
    kind: ArgKind,
    tie: TieBreak,
    dtype: DType,
    challenger: &Candidate,
    best: &Candidate,
) -> ValueExpr {
    let binary = |op, lhs: &ValueExpr, rhs: &ValueExpr| {
        ValueExpr::Binary(op, Box::new(lhs.clone()), Box::new(rhs.clone()))
    };
    let both = |lhs: ValueExpr, rhs: ValueExpr| {
        ValueExpr::Binary(BinaryOp::And, Box::new(lhs), Box::new(rhs))
    };
    let either = |lhs: ValueExpr, rhs: ValueExpr| {
        ValueExpr::Binary(BinaryOp::Or, Box::new(lhs), Box::new(rhs))
    };
    let better = match kind {
        ArgKind::Max => BinaryOp::Greater,
        ArgKind::Min => BinaryOp::Less,
    };
    let tie_order = match tie {
        TieBreak::First => BinaryOp::Less,
        TieBreak::Last => BinaryOp::Greater,
    };
    let strictly = binary(better, &challenger.value, &best.value);
    let tied = both(
        binary(BinaryOp::Equal, &challenger.value, &best.value),
        binary(tie_order, &challenger.position, &best.position),
    );
    let plain = either(strictly, tied);
    if !matches!(dtype, DType::F32 | DType::F16 | DType::Bf16) {
        return plain;
    }
    let is_nan = |value: &ValueExpr| ValueExpr::Unary(UnaryOp::IsNaN, Box::new(value.clone()));
    let earlier = binary(BinaryOp::Less, &challenger.position, &best.position);
    ValueExpr::Select {
        condition: Box::new(is_nan(&best.value)),
        then: Box::new(both(is_nan(&challenger.value), earlier)),
        otherwise: Box::new(either(is_nan(&challenger.value), plain)),
    }
}
