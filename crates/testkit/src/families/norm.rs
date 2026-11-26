//! Softmax and every normalisation, at every float type, over random axes
//! and parameter shapes.

use super::cases::single_node;
use super::values::{Domain, absolute, axis_dims, operand_dims, tensor};
use crate::compare::Tolerance;
use proptest::prelude::*;
use proptest::sample::select;
use ptxc_core::{Axis, DType, ElementCount, HostTensor};
use ptxc_graph::{
    BatchNormAttrs, F32Attr, Graph, InstanceNormAttrs, LayerNormAttrs, LayerNormKind, NormOutputs,
    Op, SoftmaxAttrs, SoftmaxKind, StashType,
};

const FLOATS: [DType; 3] = [DType::F32, DType::F16, DType::Bf16];
const EPSILON: F32Attr = F32Attr::new(1e-5);

fn floats() -> impl Strategy<Value = DType> {
    select(FLOATS.to_vec())
}

/// A moderate tensor of `dtype` and `dims`: variances stay in f32's range.
fn moderate(dtype: DType, dims: Vec<u64>) -> BoxedStrategy<HostTensor> {
    tensor(dtype, dims, Domain::Moderate)
}

/// An input a normalisation's f32 statistics handle well: normalising
/// scales each deviation by `1 / sqrt(variance + epsilon)`, which turns
/// f32's rounding of a mean into an error up to `1 / sqrt(epsilon)` times
/// larger when a block barely varies (D44). Whole numbers keep every
/// nonzero deviation large against that rounding.
fn normalized_input(dtype: DType, dims: Vec<u64>) -> BoxedStrategy<HostTensor> {
    tensor(dtype, dims, Domain::Integral)
}

/// Each output combines at most every input element.
pub fn tolerance_for(input: &HostTensor) -> impl Fn(DType) -> Tolerance {
    let length = input.shape().element_count().get().max(1);
    move |dtype| Tolerance::for_dtype(dtype).for_reduction(ElementCount::new(length))
}

fn graph_of(op: Op, inputs: Vec<HostTensor>) -> (Graph, Vec<HostTensor>) {
    (single_node(op, &inputs), inputs)
}

/// `Softmax` and `LogSoftmax` along any axis.
pub fn softmax_graphs() -> impl Strategy<Value = (Graph, Vec<HostTensor>)> {
    let kind = prop_oneof![Just(SoftmaxKind::Softmax), Just(SoftmaxKind::LogSoftmax)];
    (floats(), axis_dims(), kind)
        .prop_flat_map(|(dtype, dims, kind)| {
            let rank = dims.len();
            (tensor(dtype, dims, Domain::Any), 0..rank, Just(kind))
        })
        .prop_map(|(input, axis, kind)| {
            graph_of(Op::Softmax(SoftmaxAttrs { kind, axis: Axis::new(axis) }), vec![input])
        })
}

fn layer_norm_kinds() -> impl Strategy<Value = LayerNormKind> {
    prop_oneof![
        Just(LayerNormKind::Layer(NormOutputs::Normalized)),
        Just(LayerNormKind::Layer(NormOutputs::WithStatistics)),
        Just(LayerNormKind::Rms),
    ]
}

/// A layer or RMS normalisation with a scale (of another float type, for
/// RMS) and, for layer normalisation, sometimes a bias, each broadcasting
/// to the normalised block.
pub fn layer_norm_graphs() -> impl Strategy<Value = (Graph, Vec<HostTensor>)> {
    let stash = prop_oneof![Just(StashType::F32), Just(StashType::Bf16)];
    (floats(), floats(), axis_dims(), layer_norm_kinds(), stash, any::<bool>())
        .prop_flat_map(|(dtype, rms_scale, dims, kind, stash, has_bias)| {
            let rank = dims.len();
            (0..rank).prop_flat_map(move |axis| {
                let block = dims[axis..].to_vec();
                let scale_dtype = if kind == LayerNormKind::Rms { rms_scale } else { dtype };
                let bias_count = usize::from(has_bias && kind != LayerNormKind::Rms);
                let parameter =
                    |dtype| operand_dims(block.clone()).prop_flat_map(move |d| moderate(dtype, d));
                let attrs = LayerNormAttrs { kind, axis: Axis::new(axis), epsilon: EPSILON, stash };
                let biases = proptest::collection::vec(parameter(dtype), bias_count);
                (Just(attrs), normalized_input(dtype, dims.clone()), parameter(scale_dtype), biases)
            })
        })
        .prop_map(|(attrs, input, scale, biases)| {
            let inputs = [vec![input, scale], biases].concat();
            graph_of(Op::LayerNorm(attrs), inputs)
        })
}

/// The channel count of an `[N, C, ...]` shape; a 1-D shape has one.
pub const fn channels(dims: &[u64]) -> u64 {
    if let [_, channels, ..] = dims { *channels } else { 1 }
}

/// Inference batch normalisation with scale and bias of one float type
/// and mean and variance of another, the variance never negative.
pub fn batch_norm_graphs() -> impl Strategy<Value = (Graph, Vec<HostTensor>)> {
    (floats(), floats(), floats(), axis_dims())
        .prop_flat_map(|(dtype, affine, statistics, dims)| {
            let per_channel = vec![channels(&dims)];
            let variance = moderate(statistics, per_channel.clone()).prop_map(|v| absolute(&v));
            (
                moderate(dtype, dims),
                moderate(affine, per_channel.clone()),
                moderate(affine, per_channel.clone()),
                moderate(statistics, per_channel),
                variance,
            )
        })
        .prop_map(|(x, scale, bias, mean, variance)| {
            let attrs = BatchNormAttrs { epsilon: EPSILON };
            graph_of(Op::BatchNorm(attrs), vec![x, scale, bias, mean, variance])
        })
}

/// `InstanceNormalization` over every spatial rank.
pub fn instance_norm_graphs() -> impl Strategy<Value = (Graph, Vec<HostTensor>)> {
    let dims = proptest::collection::vec(prop_oneof![1 => Just(0_u64), 12 => 1_u64..=3], 3..=4);
    (floats(), dims)
        .prop_flat_map(|(dtype, dims)| {
            let per_channel = vec![dims[1]];
            (
                normalized_input(dtype, dims),
                moderate(dtype, per_channel.clone()),
                moderate(dtype, per_channel),
            )
        })
        .prop_map(|(x, scale, bias)| {
            graph_of(Op::InstanceNorm(InstanceNormAttrs { epsilon: EPSILON }), vec![x, scale, bias])
        })
}
