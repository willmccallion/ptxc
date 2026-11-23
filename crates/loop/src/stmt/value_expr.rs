//! `ValueExpr`: typed scalar computations on data, kept apart from index expressions.

use super::buffer::{Buffer, BufferId};
use crate::compute::{BinaryOp, Scalar, UnaryOp};
use crate::expr::ExprId;
use ptxc_core::float::Overflow;
use ptxc_core::{DType, IndexVec};

/// A scalar computed from buffers and constants. Index arithmetic stays in
/// the kernel's `ExprArena`; data reaches an index only through a kernel's
/// `LoadIndex` (D4).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum ValueExpr {
    /// An element of a buffer.
    Load {
        /// The buffer.
        buffer: BufferId,
        /// The element's flat position.
        index: ExprId,
    },
    /// A constant.
    Const(Scalar),
    /// An index's value as data, such as an arg-reduction's position.
    Index {
        /// The index.
        index: ExprId,
        /// The integer type it becomes.
        dtype: DType,
    },
    /// An operation on one value.
    Unary(UnaryOp, Box<Self>),
    /// An operation on two values of one type.
    Binary(BinaryOp, Box<Self>, Box<Self>),
    /// A conversion to another element type.
    Cast {
        /// The element type converted to.
        to: DType,
        /// What an out-of-range value becomes, for FP8 targets.
        overflow: Overflow,
        /// The value converted.
        value: Box<Self>,
    },
    /// `then` where `condition` holds, `otherwise` elsewhere; both are
    /// evaluated.
    Select {
        /// A boolean.
        condition: Box<Self>,
        /// The value where it is true.
        then: Box<Self>,
        /// The value where it is false.
        otherwise: Box<Self>,
    },
}

impl ValueExpr {
    /// The element type, for a well-typed expression over `buffers`.
    pub fn dtype(&self, buffers: &IndexVec<BufferId, Buffer>) -> DType {
        match self {
            Self::Load { buffer, .. } => buffers[*buffer].dtype,
            Self::Const(scalar) => scalar.dtype(),
            Self::Index { dtype, .. } | Self::Cast { to: dtype, .. } => *dtype,
            Self::Unary(UnaryOp::Not | UnaryOp::IsNaN | UnaryOp::IsInf(_), _) => DType::Bool,
            Self::Unary(_, value) => value.dtype(buffers),
            Self::Binary(op, lhs, _) => {
                if is_predicate(*op) {
                    DType::Bool
                } else {
                    lhs.dtype(buffers)
                }
            }
            Self::Select { then, .. } => then.dtype(buffers),
        }
    }

    /// Whether the expression loads from `buffer` anywhere in it.
    pub fn reads(&self, buffer: BufferId) -> bool {
        match self {
            Self::Load { buffer: read, .. } => *read == buffer,
            Self::Const(_) | Self::Index { .. } => false,
            Self::Unary(_, value) | Self::Cast { value, .. } => value.reads(buffer),
            Self::Binary(_, lhs, rhs) => lhs.reads(buffer) || rhs.reads(buffer),
            Self::Select { condition, then, otherwise } => {
                condition.reads(buffer) || then.reads(buffer) || otherwise.reads(buffer)
            }
        }
    }
}

/// Whether `op` gives a boolean whatever its operands are.
pub(crate) const fn is_predicate(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Equal
            | BinaryOp::Less
            | BinaryOp::LessOrEqual
            | BinaryOp::Greater
            | BinaryOp::GreaterOrEqual
            | BinaryOp::And
            | BinaryOp::Or
            | BinaryOp::Xor
    )
}
