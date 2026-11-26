//! Softmax and batch, instance, layer and RMS normalisation, in f64.

use super::{result, round_to_output};
use crate::error::{InterpError, Malformed};
use crate::eval::first;
use crate::extents::Extents;
use crate::numeric::{count_to_f64, store_wide, wide_floats};
use ptxc_core::{HostTensor, TensorData};
use ptxc_graph::{
    BatchNormAttrs, InstanceNormAttrs, LayerNormAttrs, LayerNormKind, NormOutputs, SoftmaxAttrs,
    SoftmaxKind, TensorType,
};

/// The blocks a tensor normalises over: every axis from `axis` on, so each
/// block is that many consecutive elements.
fn blocks<'v>(
    values: &'v [f64],
    extents: &Extents,
    axis: usize,
) -> impl Iterator<Item = &'v [f64]> {
    let outer: usize = extents.dims()[..axis].iter().product();
    let inner: usize = extents.dims()[axis..].iter().product();
    (0..outer).map(move |b| &values[b * inner..(b + 1) * inner])
}

/// The mean and population variance of a block.
fn moments(block: &[f64]) -> (f64, f64) {
    let count = count_to_f64(block.len());
    let mean = block.iter().sum::<f64>() / count;
    let squares = block.iter().map(|v| {
        let deviation = v - mean;
        deviation * deviation
    });
    let variance = squares.sum::<f64>() / count;
    (mean, variance)
}

/// A parameter's elements broadcast to `block`'s shape.
fn broadcast_to(tensor: &HostTensor, block: &Extents) -> Result<Vec<f64>, InterpError> {
    let (values, own) = (wide_floats(tensor.data())?, Extents::of(tensor.shape())?);
    Ok(block.indices().map(|index| values[own.broadcast_offset(&index)]).collect())
}

/// `Softmax` or `LogSoftmax` along one axis, shifted by each row's maximum.
pub(super) fn softmax(
    attrs: &SoftmaxAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let extents = Extents::of(input.shape())?;
    let values = wide_floats(input.data())?;
    let axis = attrs.axis.index();
    let (extent, stride) = (extents.dims()[axis], extents.strides()[axis]);
    let mut out = vec![0.0; values.len()];
    for start in (0..values.len()).filter(|flat| flat / stride % extent == 0) {
        let row: Vec<usize> = (0..extent).map(|i| start + i * stride).collect();
        let max = row.iter().map(|&i| values[i]).fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|&i| (values[i] - max).exp()).sum();
        for &i in &row {
            out[i] = match attrs.kind {
                SoftmaxKind::Softmax => (values[i] - max).exp() / sum,
                SoftmaxKind::LogSoftmax => values[i] - max - sum.ln(),
            };
        }
    }
    round_to_output(&out, output)
}

/// `LayerNormalization` or `RMSNormalization` over the axes from `axis` on,
/// with the scale and bias broadcast to that block's shape.
pub(super) fn layer_norm(
    attrs: &LayerNormAttrs,
    operands: &[&HostTensor],
    outputs: &[&TensorType],
) -> Result<Vec<HostTensor>, InterpError> {
    let (input, scale, bias) = match operands {
        [x, scale] => (x, scale, None),
        [x, scale, bias] => (x, scale, Some(bias)),
        _ => return Err(Malformed::Operands.into()),
    };
    let extents = Extents::of(input.shape())?;
    let axis = attrs.axis.index();
    let block = Extents::from_dims(extents.dims()[axis..].to_vec());
    let scales = broadcast_to(scale, &block)?;
    let biases = bias
        .map(|bias| broadcast_to(bias, &block))
        .transpose()?
        .unwrap_or_else(|| vec![0.0; block.count()]);
    let values = wide_floats(input.data())?;
    let statistics: Vec<(f64, f64)> =
        blocks(&values, &extents, axis).map(|chunk| statistics(attrs, chunk)).collect();
    let normalized: Vec<f64> = blocks(&values, &extents, axis)
        .zip(&statistics)
        .flat_map(|(chunk, &(center, inverse))| {
            chunk
                .iter()
                .zip(&scales)
                .zip(&biases)
                .map(move |((v, s), b)| ((v - center) * inverse).mul_add(*s, *b))
        })
        .collect();
    let [normalized_type, statistic_types @ ..] = outputs else {
        return Err(Malformed::Operands.into());
    };
    let mut results =
        vec![result(normalized_type, round_to_output(&normalized, normalized_type)?)?];
    if attrs.kind == LayerNormKind::Layer(NormOutputs::WithStatistics) {
        let (means, inverses): (Vec<f64>, Vec<f64>) = statistics.into_iter().unzip();
        for (statistic, ty) in <[_; 2]>::from((means, inverses)).iter().zip(statistic_types) {
            results.push(result(ty, store_wide(statistic, ty.dtype())?)?);
        }
    }
    Ok(results)
}

/// A block's center and inverse deviation: the mean and `1 / sqrt(var +
/// epsilon)` for layer norm, 0 and `1 / sqrt(mean(x^2) + epsilon)` for RMS.
fn statistics(attrs: &LayerNormAttrs, block: &[f64]) -> (f64, f64) {
    let epsilon = f64::from(attrs.epsilon.get());
    match attrs.kind {
        LayerNormKind::Layer(_) => {
            let (mean, variance) = moments(block);
            (mean, 1.0 / (variance + epsilon).sqrt())
        }
        LayerNormKind::Rms => {
            let mean_square = block.iter().map(|v| v * v).sum::<f64>() / count_to_f64(block.len());
            (0.0, 1.0 / (mean_square + epsilon).sqrt())
        }
    }
}

/// The channel count and the elements per channel per sample of an
/// `[N, C, …]` input; a 1-D input has one channel.
fn channels(extents: &Extents) -> (usize, usize) {
    match extents.dims() {
        [_, channels, spatial @ ..] => (*channels, spatial.iter().product()),
        _ => (1, 1),
    }
}

/// Inference `BatchNormalization`: `(x - mean) / sqrt(var + epsilon) *
/// scale + bias` per channel.
pub(super) fn batch_norm(
    attrs: BatchNormAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let [x, scale, bias, mean, variance] = operands else { return Err(Malformed::Operands.into()) };
    let (channel_count, inner) = channels(&Extents::of(x.shape())?);
    let [scale, bias, mean, variance] =
        [scale, bias, mean, variance].map(|t| wide_floats(t.data()));
    let (scale, bias, mean, variance) = (scale?, bias?, mean?, variance?);
    let epsilon = f64::from(attrs.epsilon.get());
    let normalized: Vec<f64> = wide_floats(x.data())?
        .iter()
        .enumerate()
        .map(|(flat, v)| {
            let c = flat / inner % channel_count;
            ((v - mean[c]) / (variance[c] + epsilon).sqrt()).mul_add(scale[c], bias[c])
        })
        .collect();
    round_to_output(&normalized, output)
}

/// `InstanceNormalization`: each sample's channel normalised over its
/// spatial axes, then scaled and shifted per channel.
pub(super) fn instance_norm(
    attrs: InstanceNormAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let [x, scale, bias] = operands else { return Err(Malformed::Operands.into()) };
    let extents = Extents::of(x.shape())?;
    let (channel_count, _) = channels(&extents);
    let (scale, bias, values) =
        (wide_floats(scale.data())?, wide_floats(bias.data())?, wide_floats(x.data())?);
    let (scale, bias) = (&scale, &bias);
    let epsilon = f64::from(attrs.epsilon.get());
    let normalized: Vec<f64> = blocks(&values, &extents, 2.min(extents.rank()))
        .enumerate()
        .flat_map(|(plane, block)| {
            let c = plane % channel_count;
            let (mean, variance) = moments(block);
            let inverse = 1.0 / (variance + epsilon).sqrt();
            block.iter().map(move |v| ((v - mean) * inverse).mul_add(scale[c], bias[c]))
        })
        .collect();
    round_to_output(&normalized, output)
}
