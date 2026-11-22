//! Convolution attributes and type rules.

use super::accepts::Accepts;
use super::window::{Rounding, WindowAxis, output_extent, split_leading_two};
use super::{TypeError, TypeRule, operands_with_optional, require_dtype};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::{DType, Dim, Rank, Shape};
use std::num::NonZeroU64;

/// The number of groups a convolution splits its channels into.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Groups(NonZeroU64);

impl Groups {
    /// The count, or `None` for zero.
    pub const fn new(count: u64) -> Option<Self> {
        match NonZeroU64::new(count) {
            Some(count) => Some(Self(count)),
            None => None,
        }
    }

    /// The count.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// ONNX `Conv`: `X: [N, C, D1, ...]` and weights `W: [M, C / groups, K1, ...]`
/// give `[N, M, O1, ...]`, with an optional bias `B: [M]`.
///
/// The kernel is `W`'s spatial dims, so there is no second copy of it here to
/// disagree with.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ConvAttrs {
    /// One entry per spatial axis.
    pub axes: Vec<WindowAxis>,
    /// Channel groups; 1 is an ordinary convolution.
    pub groups: Groups,
}

impl TypeRule for ConvAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let ([input, weights], bias) = operands_with_optional(operands)?;
        Accepts::Float.check(input.dtype())?;
        require_dtype(input.dtype(), weights)?;
        let shape = self.output_shape(input.shape(), weights.shape())?;
        if let Some(bias) = bias {
            check_bias(bias, input.dtype(), shape.dims()[1])?;
        }
        Ok(vec![TensorType::new(input.dtype(), shape, Layout::RowMajor)])
    }
}

impl ConvAttrs {
    /// The output shape for an input and weights of these shapes, whatever
    /// their element types.
    pub(super) fn output_shape(&self, input: &Shape, weights: &Shape) -> Result<Shape, TypeError> {
        let rank = Rank::new(self.axes.len() + 2);
        let (batch, channels, spatial) = split_leading_two(input, rank)?;
        let (filters, group_channels, kernel) = split_leading_two(weights, rank)?;
        self.check_channels(channels, filters, group_channels)?;
        let mut dims = vec![batch, filters];
        for ((&extent, &kernel), &axis) in spatial.iter().zip(kernel).zip(&self.axes) {
            dims.push(output_extent(extent, kernel, axis, Rounding::Floor)?);
        }
        Ok(Shape::new(dims)?)
    }

    /// `C` must be `W`'s channels times the groups, and the groups must
    /// divide `M`.
    fn check_channels(
        &self,
        channels: Dim,
        filters: Dim,
        group_channels: Dim,
    ) -> Result<(), TypeError> {
        let groups = self.groups.get();
        let expected = group_channels.get().checked_mul(groups);
        if expected != Some(channels.get()) {
            return Err(TypeError::Channels {
                input: channels,
                weights: group_channels,
                groups: self.groups,
            });
        }
        if !filters.get().is_multiple_of(groups) {
            return Err(TypeError::FiltersNotDivisible { filters, groups: self.groups });
        }
        Ok(())
    }
}

pub(super) fn check_bias(bias: &TensorType, dtype: DType, filters: Dim) -> Result<(), TypeError> {
    require_dtype(dtype, bias)?;
    match *bias.shape().dims() {
        [length] if length == filters => Ok(()),
        [length] => Err(TypeError::ParameterLength { expected: filters, actual: length }),
        _ => Err(TypeError::Rank { expected: Rank::new(1), actual: bias.shape().rank() }),
    }
}
