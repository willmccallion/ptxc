//! `ComputeError`: why a compute definition is not well-formed.

use super::iter_var::IterVarId;
use super::names::{TensorName, VarName};
use super::tensor_expr::TensorParamId;
use crate::expr::VarId;
use ptxc_core::{DType, Rank, ShapeError};
use thiserror::Error;

/// A compute definition `ComputeBuilder::finish` rejects.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ComputeError {
    /// No outputs were added.
    #[error("a compute definition needs at least one output")]
    NoOutputs,
    /// A tensor id that is not one of the definition's inputs.
    #[error("tensor {0:?} is not an input of this definition")]
    UnknownTensor(TensorParamId),
    /// An index variable that is not one of the definition's.
    #[error("index variable {0:?} is not defined")]
    UnknownVar(VarId),
    /// An axis that is not one of the definition's.
    #[error("axis {0:?} is not defined")]
    UnknownAxis(IterVarId),
    /// A load or loaded index with the wrong number of indices.
    #[error("{tensor} has rank {}, but is indexed with {} indices", .expected.get(), .actual.get())]
    IndexArity {
        /// The tensor.
        tensor: TensorName,
        /// Its rank.
        expected: Rank,
        /// The indices given.
        actual: Rank,
    },
    /// An index read from a tensor that does not hold integers.
    #[error("index tensor {tensor} holds {dtype:?}, not i32 or i64")]
    IndexNotInteger {
        /// The tensor.
        tensor: TensorName,
        /// Its element type.
        dtype: DType,
    },
    /// A reduction axis read outside the reduction over it.
    #[error("reduction axis {0} is read outside its reduction")]
    ReduceAxisOutOfScope(VarName),
    /// A spatial axis named as a reduction axis.
    #[error("{0} is a spatial axis, not a reduction axis")]
    NotAReduceAxis(VarName),
    /// A reduction axis reduced over twice.
    #[error("reduction axis {0} is reduced over twice")]
    ReduceAxisReused(VarName),
    /// A reduction axis no reduction uses.
    #[error("reduction axis {0} is never reduced over")]
    UnusedReduceAxis(VarName),
    /// A reduction with no axes.
    #[error("a reduction needs at least one axis")]
    EmptyReduce,
    /// An operation given an element type it does not take.
    #[error("{op} does not take {dtype:?}")]
    UnsupportedDType {
        /// The operation.
        op: &'static str,
        /// The element type.
        dtype: DType,
    },
    /// An operation given two different element types.
    #[error("{op} takes one element type, given {lhs:?} and {rhs:?}")]
    MixedDTypes {
        /// The operation.
        op: &'static str,
        /// The first operand's type.
        lhs: DType,
        /// The second operand's type.
        rhs: DType,
    },
    /// The outputs' shape has more elements than fit in `u64`.
    #[error(transparent)]
    Shape(#[from] ShapeError),
}
