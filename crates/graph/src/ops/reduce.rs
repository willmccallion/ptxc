//! Reduction attributes and type rules.

use super::accepts::Accepts;
use super::{TypeError, TypeRule, check_axis, exact_operands};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::{Axis, DType, Dim, Shape};

/// What a reduction combines elements with.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ReduceOp {
    /// `ReduceSum`.
    Sum,
    /// `ReduceMean`.
    Mean,
    /// `ReduceProd`.
    Prod,
    /// `ReduceMax`.
    Max,
    /// `ReduceMin`.
    Min,
    /// `ReduceSumSquare`: the sum of squares.
    SumSquare,
    /// `ReduceL1`: the sum of absolute values.
    L1,
    /// `ReduceL2`: the square root of the sum of squares.
    L2,
    /// `ReduceLogSum`: the log of the sum.
    LogSum,
    /// `ReduceLogSumExp`: the log of the sum of exponentials.
    LogSumExp,
}

/// Whether reduced axes stay as extent 1 or are removed, from ONNX's
/// `keepdims`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum KeepDims {
    /// Reduced axes stay, with extent 1.
    Keep,
    /// Reduced axes are removed.
    Drop,
}

/// An ONNX reduction over explicit axes. ONNX's "no axes means every axis"
/// is resolved by the importer, so `axes` lists exactly what is reduced; an
/// empty list reduces nothing.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ReduceAttrs {
    /// The combining op.
    pub op: ReduceOp,
    /// The axes reduced, each at most once.
    pub axes: Vec<Axis>,
    /// What happens to the reduced axes.
    pub keep_dims: KeepDims,
}

impl ReduceOp {
    const fn accepts(self) -> Accepts {
        match self {
            Self::Max | Self::Min => Accepts::NumericOrBool,
            _ => Accepts::FloatOrWideInt,
        }
    }
}

impl TypeRule for ReduceAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        self.op.accepts().check(input.dtype())?;
        let shape = reduced_shape(input.shape(), &self.axes, self.keep_dims)?;
        Ok(vec![TensorType::new(input.dtype(), shape, Layout::RowMajor)])
    }
}

/// Which extreme `ArgMax`/`ArgMin` look for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ArgKind {
    /// The index of the largest element.
    Max,
    /// The index of the smallest element.
    Min,
}

/// Which index wins a tie, from ONNX's `select_last_index`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TieBreak {
    /// The first occurrence.
    First,
    /// The last occurrence.
    Last,
}

/// ONNX `ArgMax` or `ArgMin` along one axis, giving `i64` indices.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ArgReduceAttrs {
    /// The extreme sought.
    pub kind: ArgKind,
    /// The axis searched.
    pub axis: Axis,
    /// What happens to the searched axis.
    pub keep_dims: KeepDims,
    /// Which index wins a tie.
    pub tie: TieBreak,
}

impl TypeRule for ArgReduceAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        Accepts::Numeric.check(input.dtype())?;
        let shape = reduced_shape(input.shape(), &[self.axis], self.keep_dims)?;
        Ok(vec![TensorType::new(DType::I64, shape, Layout::RowMajor)])
    }
}

/// `shape` with `axes` reduced, each of which must be in range and listed
/// once.
fn reduced_shape(shape: &Shape, axes: &[Axis], keep_dims: KeepDims) -> Result<Shape, TypeError> {
    for (position, &axis) in axes.iter().enumerate() {
        check_axis(shape, axis)?;
        if axes[..position].contains(&axis) {
            return Err(TypeError::RepeatedAxis(axis));
        }
    }
    let mut dims = Vec::new();
    for (axis, &dim) in shape.rank().axes().zip(shape.dims()) {
        match (axes.contains(&axis), keep_dims) {
            (false, _) => dims.push(dim),
            (true, KeepDims::Keep) => dims.push(Dim::new(1)),
            (true, KeepDims::Drop) => {}
        }
    }
    Ok(Shape::new(dims)?)
}
