//! `TensorExpr`: the scalar computation for one output element.

use super::iter_var::IterVarId;
use super::scalar::Scalar;
use super::scalar_op::{ArgKind, BinaryOp, ReduceOp, TieBreak, UnaryOp};
use crate::expr::ExprId;
use ptxc_core::DType;
use ptxc_core::float::Overflow;

ptxc_core::define_index!(
    /// An input tensor of a compute definition.
    pub TensorParamId
);

/// The value of one output element, in terms of the iteration variables.
/// Index arithmetic stays in the `ExprArena` (D4): a load's index is a list
/// of index expressions, one per axis of the tensor.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TensorExpr {
    /// An element of an input tensor.
    Load {
        /// The tensor.
        tensor: TensorParamId,
        /// One index per axis.
        index: Vec<ExprId>,
    },
    /// A constant.
    Const(Scalar),
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
    /// `then` where `condition` holds, `otherwise` elsewhere.
    Select {
        /// A boolean.
        condition: Box<Self>,
        /// The value where it is true.
        then: Box<Self>,
        /// The value where it is false.
        otherwise: Box<Self>,
    },
    /// `body` combined over every value of the reduction axes.
    Reduce {
        /// How values combine.
        op: ReduceOp,
        /// The reduction axes, each used by no other reduction.
        axes: Vec<IterVarId>,
        /// The value reduced, which may read the axes.
        body: Box<Self>,
    },
    /// The position along `axis` of `body`'s extreme value, as `i64`.
    ArgReduce {
        /// The extreme sought.
        kind: ArgKind,
        /// Which position wins a tie.
        tie: TieBreak,
        /// The reduction axis.
        axis: IterVarId,
        /// The value compared.
        body: Box<Self>,
    },
}
