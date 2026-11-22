//! `Reshape`, `Expand`, `Transpose` and `Tile`.

use crate::ops::elementwise::broadcast;
use crate::ops::{TypeError, TypeRule, check_axis, exact_operands, require_rank};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::{Axis, Dim, Rank, Shape, ShapeError};

/// `Reshape` to `target`, which must hold as many elements as the input.
/// The importer resolves ONNX's `0` and `-1` entries.
pub(in crate::ops) fn infer_reshape(
    target: &Shape,
    operands: &[&TensorType],
) -> Result<Vec<TensorType>, TypeError> {
    let [input] = exact_operands(operands)?;
    if input.shape().element_count() != target.element_count() {
        let (from, to) = (input.shape().clone(), target.clone());
        return Err(TypeError::ElementCountMismatch { from, to });
    }
    Ok(vec![TensorType::new(input.dtype(), target.clone(), Layout::RowMajor)])
}

/// ONNX `Expand`: the input broadcast with `target`, in both directions, so
/// the result may be larger than `target`.
pub(in crate::ops) fn infer_expand(
    target: &Shape,
    operands: &[&TensorType],
) -> Result<Vec<TensorType>, TypeError> {
    let [input] = exact_operands(operands)?;
    let shape = broadcast(input.shape(), target)?;
    Ok(vec![TensorType::new(input.dtype(), shape, Layout::RowMajor)])
}

/// ONNX `Transpose`: output axis `i` is input axis `permutation[i]`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TransposeAttrs {
    /// Every input axis exactly once, in output order.
    pub permutation: Vec<Axis>,
}

impl TypeRule for TransposeAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        let shape = input.shape();
        require_rank(shape, Rank::new(self.permutation.len()))?;
        let mut dims = Vec::with_capacity(self.permutation.len());
        for (position, &axis) in self.permutation.iter().enumerate() {
            check_axis(shape, axis)?;
            if self.permutation[..position].contains(&axis) {
                return Err(TypeError::RepeatedAxis(axis));
            }
            dims.push(shape.dims()[axis.index()]);
        }
        Ok(vec![TensorType::new(input.dtype(), Shape::new(dims)?, Layout::RowMajor)])
    }
}

/// How many times `Tile` repeats an axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RepeatCount(u64);

impl RepeatCount {
    /// Wraps a count.
    pub const fn new(count: u64) -> Self {
        Self(count)
    }

    /// The count.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// ONNX `Tile`: each axis repeated its count of times.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TileAttrs {
    /// One count per input axis.
    pub repeats: Vec<RepeatCount>,
}

impl TypeRule for TileAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        let shape = input.shape();
        require_rank(shape, Rank::new(self.repeats.len()))?;
        let overflow = || ShapeError::ElementCountOverflow { dims: shape.dims().to_vec() };
        let dims = shape
            .dims()
            .iter()
            .zip(&self.repeats)
            .map(|(dim, repeat)| {
                dim.get().checked_mul(repeat.get()).map(Dim::new).ok_or_else(overflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(vec![TensorType::new(input.dtype(), Shape::new(dims)?, Layout::RowMajor)])
    }
}
