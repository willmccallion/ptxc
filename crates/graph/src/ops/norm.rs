//! Softmax and normalisation attributes and type rules.

use super::accepts::Accepts;
use super::attr::F32Attr;
use super::elementwise::broadcast_to;
use super::{
    TypeError, TypeRule, check_axis, exact_operands, operands_with_optional, require_dtype,
};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::{Axis, DType, Dim, Rank, Shape};

/// Which softmax ONNX op.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SoftmaxKind {
    /// `Softmax`: `e^x / sum(e^x)`.
    Softmax,
    /// `LogSoftmax`: the log of `Softmax`.
    LogSoftmax,
}

/// ONNX `Softmax` or `LogSoftmax` along one axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SoftmaxAttrs {
    /// Which softmax.
    pub kind: SoftmaxKind,
    /// The axis normalised over.
    pub axis: Axis,
}

impl TypeRule for SoftmaxAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        Accepts::Float.check(input.dtype())?;
        check_axis(input.shape(), self.axis)?;
        Ok(vec![TensorType::new(input.dtype(), input.shape().clone(), Layout::RowMajor)])
    }
}

/// The element type statistics are computed and stored in, from ONNX's
/// `stash_type`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StashType {
    /// f32, ONNX's default.
    F32,
    /// bf16.
    Bf16,
}

impl StashType {
    const fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Bf16 => DType::Bf16,
        }
    }
}

/// Which outputs `LayerNormalization` produces.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum NormOutputs {
    /// The normalised tensor only.
    Normalized,
    /// The normalised tensor, the mean and the inverse standard deviation.
    WithStatistics,
}

/// ONNX `LayerNormalization` (operands `X`, `Scale` and an optional `B`) or
/// `RMSNormalization` (`X` and `scale`), normalising over every axis from
/// `axis` on. Scale and bias broadcast to that normalised shape.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LayerNormAttrs {
    /// Layer or RMS normalisation.
    pub kind: LayerNormKind,
    /// The first normalised axis.
    pub axis: Axis,
    /// Added to the variance before its square root.
    pub epsilon: F32Attr,
    /// The precision of the statistics.
    pub stash: StashType,
}

/// Which layer normalisation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LayerNormKind {
    /// `LayerNormalization`: subtract the mean, divide by the standard
    /// deviation.
    Layer(NormOutputs),
    /// `RMSNormalization`: divide by the root mean square. The result has
    /// the scale's element type, which may differ from the input's.
    Rms,
}

impl TypeRule for LayerNormAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let (input, scale, bias) = match self.kind {
            LayerNormKind::Layer(_) => {
                let ([input, scale], bias) = operands_with_optional(operands)?;
                (input, scale, bias)
            }
            LayerNormKind::Rms => {
                let [input, scale] = exact_operands(operands)?;
                (input, scale, None)
            }
        };
        Accepts::Float.check(input.dtype())?;
        Accepts::Float.check(scale.dtype())?;
        check_axis(input.shape(), self.axis)?;
        let normalized = Shape::new(input.shape().dims()[self.axis.index()..].to_vec())?;
        for parameter in [Some(scale), bias].into_iter().flatten() {
            broadcast_to(parameter.shape(), &normalized)?;
        }
        match self.kind {
            LayerNormKind::Rms => {
                Ok(vec![TensorType::new(scale.dtype(), input.shape().clone(), Layout::RowMajor)])
            }
            LayerNormKind::Layer(outputs) => {
                require_dtype(input.dtype(), scale)?;
                if let Some(bias) = bias {
                    require_dtype(input.dtype(), bias)?;
                }
                self.layer_outputs(input, outputs)
            }
        }
    }
}

impl LayerNormAttrs {
    fn layer_outputs(
        &self,
        input: &TensorType,
        outputs: NormOutputs,
    ) -> Result<Vec<TensorType>, TypeError> {
        let normalized = TensorType::new(input.dtype(), input.shape().clone(), Layout::RowMajor);
        if outputs == NormOutputs::Normalized {
            return Ok(vec![normalized]);
        }
        let (kept, reduced) = input.shape().dims().split_at(self.axis.index());
        let ones = reduced.iter().map(|_| Dim::new(1));
        let statistics_shape = Shape::new(kept.iter().copied().chain(ones).collect())?;
        let statistics = TensorType::new(self.stash.dtype(), statistics_shape, Layout::RowMajor);
        Ok(vec![normalized, statistics.clone(), statistics])
    }
}

/// ONNX `BatchNormalization` in inference mode: `X: [N, C, ...]` with
/// per-channel scale, bias, mean and variance. The training-mode outputs
/// are not supported.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BatchNormAttrs {
    /// Added to the variance before its square root.
    pub epsilon: F32Attr,
}

impl TypeRule for BatchNormAttrs {
    /// Scale and bias share one float type, and mean and variance another;
    /// either may differ from the input's, as ONNX's `T1` and `T2`.
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input, scale, bias, mean, variance] = exact_operands(operands)?;
        let channels = channels(input)?;
        for (parameter, partner) in [(scale, bias), (mean, variance)] {
            Accepts::Float.check(parameter.dtype())?;
            require_dtype(parameter.dtype(), partner)?;
        }
        for parameter in [scale, bias, mean, variance] {
            check_per_channel(parameter, channels)?;
        }
        Ok(vec![TensorType::new(input.dtype(), input.shape().clone(), Layout::RowMajor)])
    }
}

/// ONNX `InstanceNormalization`: `X: [N, C, D1, ...]` normalised over its
/// spatial axes per sample and channel, with per-channel scale and bias of
/// `X`'s type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InstanceNormAttrs {
    /// Added to the variance before its square root.
    pub epsilon: F32Attr,
}

impl TypeRule for InstanceNormAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input, scale, bias] = exact_operands(operands)?;
        Accepts::Float.check(input.dtype())?;
        let &[_, channels, _, ..] = input.shape().dims() else {
            return Err(TypeError::RankBelow {
                minimum: Rank::new(3),
                actual: input.shape().rank(),
            });
        };
        for parameter in [scale, bias] {
            require_dtype(input.dtype(), parameter)?;
            check_per_channel(parameter, channels)?;
        }
        Ok(vec![TensorType::new(input.dtype(), input.shape().clone(), Layout::RowMajor)])
    }
}

/// The channel count of a float `[N, C, ...]` input; ONNX treats a 1-D input
/// as having one channel.
fn channels(input: &TensorType) -> Result<Dim, TypeError> {
    Accepts::Float.check(input.dtype())?;
    match input.shape().dims() {
        [] => Err(TypeError::RankBelow { minimum: Rank::new(1), actual: Rank::new(0) }),
        [_] => Ok(Dim::new(1)),
        [_, channels, ..] => Ok(*channels),
    }
}

/// Checks that a per-channel parameter is `[channels]`.
fn check_per_channel(parameter: &TensorType, channels: Dim) -> Result<(), TypeError> {
    match *parameter.shape().dims() {
        [length] if length == channels => Ok(()),
        [length] => Err(TypeError::ParameterLength { expected: channels, actual: length }),
        _ => Err(TypeError::Rank { expected: Rank::new(1), actual: parameter.shape().rank() }),
    }
}
