//! `ComputeError`: why a compute definition is not well-formed.

use super::iter_var::IterVarId;
use super::names::{TensorName, VarName};
use super::tensor_expr::TensorParamId;
use crate::expr::VarId;
use ptxc_core::{Axis, DType, Dim, Rank, ShapeError};
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
    /// An axis listed twice in one reduction.
    #[error("reduction axis {0} is listed twice")]
    ReduceAxisReused(VarName),
    /// A second reduction; each compute definition is one stage with at
    /// most one (D40), which its output may read more than once (D80).
    #[error("a compute definition holds at most one reduction")]
    MultipleReductions,
    /// Several outputs alongside a reduction.
    #[error("a compute definition with a reduction has exactly one output")]
    ReductionWithSeveralOutputs,
    /// A load index that can leave its axis over the iteration domain.
    #[error("index {} of {tensor} can leave [0, {})", .axis.index(), .extent.get())]
    IndexOutOfBounds {
        /// The tensor.
        tensor: TensorName,
        /// The axis indexed.
        axis: Axis,
        /// The axis's extent.
        extent: Dim,
    },
    /// A load whose padding is not of its tensor's element type.
    #[error("{tensor} holds {} but is padded with {}", .expected.name(), .actual.name())]
    PaddingType {
        /// The tensor.
        tensor: TensorName,
        /// Its element type.
        expected: DType,
        /// The padding's.
        actual: DType,
    },
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
    /// An extent or tensor size past `i64::MAX`, which index arithmetic
    /// cannot hold.
    #[error("{} is too large for an index", .0.get())]
    ExtentTooLarge(Dim),
    /// The outputs' shape has more elements than fit in `u64`.
    #[error(transparent)]
    Shape(#[from] ShapeError),
}
