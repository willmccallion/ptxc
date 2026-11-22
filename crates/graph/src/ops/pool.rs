//! Pooling attributes and type rules.

use super::accepts::Accepts;
use super::window::{Rounding, WindowAxis, output_extent, split_leading_two};
use super::{TypeError, TypeRule, exact_operands};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::{Dim, Rank, Shape};

/// What a pooling window computes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PoolKind {
    /// The largest element, as ONNX `MaxPool`. Padding never wins.
    Max,
    /// The mean, as ONNX `AveragePool`.
    Average(PadCounting),
}

/// Whether `AveragePool` divides by the padding it covers, from ONNX's
/// `count_include_pad`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PadCounting {
    /// Divide by the real elements only.
    Exclude,
    /// Divide by the whole window, padding included.
    Include,
}

/// One spatial axis of a pooling window.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PoolAxis {
    /// The window's extent.
    pub kernel: Dim,
    /// How the window moves.
    pub window: WindowAxis,
}

/// ONNX `MaxPool` or `AveragePool` on `[N, C, D1, ...]`, giving `[N, C, O1,
/// ...]`. Only the pooled values are produced: `MaxPool`'s optional indices
/// output is not supported.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PoolAttrs {
    /// What each window computes.
    pub kind: PoolKind,
    /// One entry per spatial axis.
    pub axes: Vec<PoolAxis>,
    /// How a partial last window is treated.
    pub rounding: Rounding,
}

impl PoolAttrs {
    const fn accepts(&self) -> Accepts {
        match self.kind {
            PoolKind::Max => Accepts::FloatOrInt8,
            PoolKind::Average(_) => Accepts::Float,
        }
    }
}

impl TypeRule for PoolAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        self.accepts().check(input.dtype())?;
        let rank = Rank::new(self.axes.len() + 2);
        let (batch, channels, spatial) = split_leading_two(input.shape(), rank)?;
        let mut dims = vec![batch, channels];
        for (&extent, axis) in spatial.iter().zip(&self.axes) {
            dims.push(output_extent(extent, axis.kernel, axis.window, self.rounding)?);
        }
        Ok(vec![TensorType::new(input.dtype(), Shape::new(dims)?, Layout::RowMajor)])
    }
}

/// What a global pool computes over all spatial axes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GlobalPoolKind {
    /// The largest element, as ONNX `GlobalMaxPool`.
    Max,
    /// The mean, as ONNX `GlobalAveragePool`.
    Average,
}

impl TypeRule for GlobalPoolKind {
    /// `[N, C, D1, ...]` gives `[N, C, 1, ...]`, keeping every axis.
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        Accepts::Float.check(input.dtype())?;
        let [batch, channels, spatial @ ..] = input.shape().dims() else {
            let actual = input.shape().rank();
            return Err(TypeError::RankBelow { minimum: Rank::new(2), actual });
        };
        let ones = spatial.iter().map(|_| Dim::new(1));
        let dims = [*batch, *channels].into_iter().chain(ones).collect();
        Ok(vec![TensorType::new(input.dtype(), Shape::new(dims)?, Layout::RowMajor)])
    }
}
