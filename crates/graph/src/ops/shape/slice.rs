//! `Slice`, `Split` and `Concat`.

use crate::ops::{TypeError, TypeRule, check_axis, exact_operands, require_dtype};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::{Axis, Dim, Shape, ShapeError};
use std::num::{NonZeroI64, NonZeroU8};

/// The distance between consecutive elements a slice takes; negative walks
/// backwards.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SliceStep(NonZeroI64);

impl SliceStep {
    /// The step, or `None` for zero, which ONNX does not allow.
    pub const fn new(step: i64) -> Option<Self> {
        match NonZeroI64::new(step) {
            Some(step) => Some(Self(step)),
            None => None,
        }
    }

    /// A step of one: every element, in order.
    pub fn one() -> Self {
        Self(NonZeroI64::from(NonZeroU8::MIN))
    }

    /// The step.
    pub const fn get(self) -> i64 {
        self.0.get()
    }
}

/// The elements a slice takes along one axis: `length` of them, from index
/// `start`, `step` apart. The importer resolves ONNX's negative and clamped
/// `starts` and `ends` into this form once.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SliceAxis {
    /// The axis sliced.
    pub axis: Axis,
    /// The index of the first element taken.
    pub start: u64,
    /// The distance between taken elements.
    pub step: SliceStep,
    /// How many elements are taken.
    pub length: Dim,
}

/// ONNX `Slice`. Axes not listed are kept whole.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SliceAttrs {
    /// The sliced axes, each at most once.
    pub axes: Vec<SliceAxis>,
}

impl TypeRule for SliceAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        let mut dims = input.shape().dims().to_vec();
        for (position, slice) in self.axes.iter().enumerate() {
            check_axis(input.shape(), slice.axis)?;
            if self.axes[..position].iter().any(|earlier| earlier.axis == slice.axis) {
                return Err(TypeError::RepeatedAxis(slice.axis));
            }
            let extent = &mut dims[slice.axis.index()];
            if !slice.stays_within(*extent) {
                return Err(TypeError::SliceOutOfBounds(slice.axis));
            }
            *extent = slice.length;
        }
        Ok(vec![TensorType::new(input.dtype(), Shape::new(dims)?, Layout::RowMajor)])
    }
}

impl SliceAxis {
    /// Whether every element taken is an index of an axis of `extent`.
    fn stays_within(&self, extent: Dim) -> bool {
        let Some(last_offset) = self.length.get().checked_sub(1) else {
            return true;
        };
        let start = i128::from(self.start);
        let last = start + i128::from(last_offset) * i128::from(self.step.get());
        let extent = i128::from(extent.get());
        (0..extent).contains(&start) && (0..extent).contains(&last)
    }
}

/// ONNX `Split`: consecutive pieces of `axis` with the given sizes, one
/// output each. The importer resolves `num_outputs` into sizes.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SplitAttrs {
    /// The axis split.
    pub axis: Axis,
    /// Each piece's extent; they sum to the axis's extent.
    pub sizes: Vec<Dim>,
}

impl TypeRule for SplitAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        check_axis(input.shape(), self.axis)?;
        let extent = input.shape().dims()[self.axis.index()];
        let total = self.sizes.iter().try_fold(0_u64, |sum, size| sum.checked_add(size.get()));
        if self.sizes.is_empty() || total != Some(extent.get()) {
            return Err(TypeError::SplitSizes { extent, sizes: self.sizes.clone() });
        }
        self.sizes
            .iter()
            .map(|&size| {
                let mut dims = input.shape().dims().to_vec();
                dims[self.axis.index()] = size;
                Ok(TensorType::new(input.dtype(), Shape::new(dims)?, Layout::RowMajor))
            })
            .collect()
    }
}

/// ONNX `Concat` along `axis`: operands of one element type and rank that
/// agree on every other axis.
pub(in crate::ops) fn infer_concat(
    axis: Axis,
    operands: &[&TensorType],
) -> Result<Vec<TensorType>, TypeError> {
    let Some((first, rest)) = operands.split_first() else {
        return Err(TypeError::NoOperands);
    };
    check_axis(first.shape(), axis)?;
    let mut dims = first.shape().dims().to_vec();
    for operand in rest {
        check_concat_operand(first, operand, axis)?;
        let joined =
            dims[axis.index()].get().checked_add(operand.shape().dims()[axis.index()].get());
        let overflow = || ShapeError::ElementCountOverflow { dims: dims.clone() };
        dims[axis.index()] = Dim::new(joined.ok_or_else(overflow)?);
    }
    Ok(vec![TensorType::new(first.dtype(), Shape::new(dims)?, Layout::RowMajor)])
}

/// Checks that `operand` has `first`'s element type, rank and dims on every
/// axis but `axis`.
fn check_concat_operand(
    first: &TensorType,
    operand: &TensorType,
    axis: Axis,
) -> Result<(), TypeError> {
    require_dtype(first.dtype(), operand)?;
    let rank = first.shape().rank();
    if operand.shape().rank() != rank {
        return Err(TypeError::Rank { expected: rank, actual: operand.shape().rank() });
    }
    let pairs = first.shape().dims().iter().zip(operand.shape().dims());
    for (other_axis, (&expected, &actual)) in rank.axes().zip(pairs) {
        if other_axis != axis && expected != actual {
            return Err(TypeError::DimMismatch { axis: other_axis, expected, actual });
        }
    }
    Ok(())
}
