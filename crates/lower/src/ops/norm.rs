//! Softmax and batch, instance, layer and RMS normalisation: one stage per
//! statistic (D40), with statistics in f32 intermediates (D43).

use super::NodeContext;
use super::reduce::{load_reduced, wide};
use crate::builder::{GroupBuilder, StageBuilder};
use crate::compose::{add, div, f32, mul, sub, unary};
use crate::error::LowerError;
use crate::precision::{convert, demote};
use crate::stage::{Binding, IntermediateId, LoweredGroup};
use ptxc_core::{Axis, DType, Dim, Shape};
use ptxc_graph::{
    BatchNormAttrs, F32Attr, InstanceNormAttrs, KeepDims, LayerNormAttrs, LayerNormKind,
    NormOutputs, SoftmaxAttrs, SoftmaxKind, ValueId,
};
use ptxc_loop::{ExprId, ReduceOp, TensorExpr, UnaryOp};

/// `Softmax` or `LogSoftmax` in three stages: each row's maximum `m`, the
/// row's sum `s` of `e^(x - m)`, then `e^(x - m) / s` or `x - m - ln(s)`.
pub(super) fn softmax(
    cx: &NodeContext<'_>,
    attrs: SoftmaxAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    let Some(output) = cx.needed_output() else { return Ok(LoweredGroup::default()) };
    let shape = cx.graph.value(input).ty().shape();
    let rows = collapsed(cx, shape, attrs.axis.index()..=attrs.axis.index())?;
    let axis = [attrs.axis];
    let mut group = cx.group();

    let max = group.intermediate(&rows);
    let mut stage = group.stage(cx.named("max"), &rows);
    let (x, over) = load_reduced(cx, &mut stage, input, &axis, KeepDims::Keep)?;
    stage.output(Binding::Intermediate(max), over.apply(ReduceOp::Max, wide(&x)));
    push(cx, &mut group, stage)?;

    let sum = group.intermediate(&rows);
    let mut stage = group.stage(cx.named("sum"), &rows);
    let (x, over) = load_reduced(cx, &mut stage, input, &axis, KeepDims::Keep)?;
    let m = read_whole(&mut stage, max);
    stage.output(Binding::Intermediate(sum), over.sum(unary(UnaryOp::Exp, sub(wide(&x), m))));
    push(cx, &mut group, stage)?;

    let mut stage = group.stage(cx.name(), shape);
    let x = stage.broadcast(input);
    let row = statistic_index(&mut stage, attrs.axis.index()..=attrs.axis.index());
    let m = stage.load(Binding::Intermediate(max), row.clone()).expr;
    let s = stage.load(Binding::Intermediate(sum), row).expr;
    let shifted = sub(wide(&x), m);
    let result = match attrs.kind {
        SoftmaxKind::Softmax => div(unary(UnaryOp::Exp, shifted), s),
        SoftmaxKind::LogSoftmax => sub(shifted, unary(UnaryOp::Log, s)),
    };
    stage.output(Binding::Value(output), demote(result, x.dtype));
    push(cx, &mut group, stage)?;
    Ok(group.finish())
}

/// What a normalisation subtracts before scaling.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Centering {
    /// The mean, as layer and instance normalisation do.
    Mean,
    /// Nothing, as RMS normalisation does.
    None,
}

/// A normalisation's statistics: f32 intermediates with one element per
/// normalised block.
struct Statistics {
    mean: Option<IntermediateId>,
    inverse: IntermediateId,
    shape: Shape,
}

/// The statistics of `input`'s blocks over every axis from `first` on: the
/// mean, if centring, and `1 / sqrt(variance + epsilon)`, where the
/// variance is about the mean or, without centring, about 0.
fn statistics(
    cx: &NodeContext<'_>,
    group: &mut GroupBuilder<'_>,
    input: ValueId,
    first: usize,
    (centering, epsilon): (Centering, F32Attr),
) -> Result<Statistics, LowerError> {
    let shape = cx.graph.value(input).ty().shape();
    let blocks = collapsed(cx, shape, first..shape.dims().len())?;
    let axes: Vec<Axis> = (first..shape.dims().len()).map(Axis::new).collect();
    let mean = match centering {
        Centering::Mean => Some(block_mean(cx, group, input, &axes, &blocks)?),
        Centering::None => None,
    };
    let inverse = group.intermediate(&blocks);
    let mut stage = group.stage(cx.named("inverse"), &blocks);
    let (x, over) = load_reduced(cx, &mut stage, input, &axes, KeepDims::Keep)?;
    let deviation =
        mean.map_or_else(|| wide(&x), |mean| sub(wide(&x), read_whole(&mut stage, mean)));
    let variance = div(over.sum(mul(deviation.clone(), deviation)), f32(over.count.to_f32()));
    let root = unary(UnaryOp::Sqrt, add(variance, f32(epsilon.get())));
    stage.output(Binding::Intermediate(inverse), div(f32(1.0), root));
    push(cx, group, stage)?;
    Ok(Statistics { mean, inverse, shape: blocks })
}

/// A stage computing each block's mean into a new intermediate.
fn block_mean(
    cx: &NodeContext<'_>,
    group: &mut GroupBuilder<'_>,
    input: ValueId,
    axes: &[Axis],
    blocks: &Shape,
) -> Result<IntermediateId, LowerError> {
    let mean = group.intermediate(blocks);
    let mut stage = group.stage(cx.named("mean"), blocks);
    let (x, over) = load_reduced(cx, &mut stage, input, axes, KeepDims::Keep)?;
    stage.output(Binding::Intermediate(mean), div(over.sum(wide(&x)), f32(over.count.to_f32())));
    push(cx, group, stage)?;
    Ok(mean)
}

/// `(x - center) * inverse` as f32, reading the statistics of the block
/// holding the stage's output element.
fn normalized(
    stage: &mut StageBuilder<'_>,
    x: &TensorExpr,
    statistics: &Statistics,
    first: usize,
) -> TensorExpr {
    let block = statistic_index(stage, first..);
    let inverse = stage.load(Binding::Intermediate(statistics.inverse), block.clone()).expr;
    let centered = statistics.mean.map_or_else(
        || x.clone(),
        |mean| sub(x.clone(), stage.load(Binding::Intermediate(mean), block).expr),
    );
    mul(centered, inverse)
}

/// `LayerNormalization` or `RMSNormalization`: statistics over every axis
/// from `axis` on, then the normalised tensor scaled (and shifted) by
/// parameters broadcast to the normalised block, then any statistics
/// outputs.
pub(super) fn layer_norm(
    cx: &NodeContext<'_>,
    attrs: LayerNormAttrs,
) -> Result<LoweredGroup, LowerError> {
    let (input, scale, bias) = match cx.operands()[..] {
        [input, scale] => (input, scale, None),
        [input, scale, bias] => (input, scale, Some(bias)),
        _ => return Err(LowerError::Operands(cx.id)),
    };
    let centering =
        if attrs.kind == LayerNormKind::Rms { Centering::None } else { Centering::Mean };
    let first = attrs.axis.index();
    let mut group = cx.group();
    let statistics = statistics(cx, &mut group, input, first, (centering, attrs.epsilon))?;
    let outputs: Vec<ValueId> = cx.node.outputs().iter().copied().collect();
    if let Some(&output) = outputs.first().filter(|value| cx.needed.contains(value)) {
        let shape = cx.graph.value(input).ty().shape();
        let mut stage = group.stage(cx.name(), shape);
        let x = wide(&stage.broadcast(input));
        let scaled =
            mul(normalized(&mut stage, &x, &statistics, first), wide(&stage.broadcast(scale)));
        let result = match bias {
            Some(bias) => add(scaled, wide(&stage.broadcast(bias))),
            None => scaled,
        };
        let dtype = cx.graph.value(output).ty().dtype();
        stage.output(Binding::Value(output), convert(result, DType::F32, dtype));
        push(cx, &mut group, stage)?;
    }
    if attrs.kind == LayerNormKind::Layer(NormOutputs::WithStatistics) {
        statistics_outputs(cx, &mut group, &statistics, outputs.get(1..).unwrap_or_default())?;
    }
    Ok(group.finish())
}

/// A stage writing the needed ones of `LayerNormalization`'s mean and
/// inverse standard deviation outputs, in their stash type.
fn statistics_outputs(
    cx: &NodeContext<'_>,
    group: &mut GroupBuilder<'_>,
    statistics: &Statistics,
    outputs: &[ValueId],
) -> Result<(), LowerError> {
    let sources = [statistics.mean, Some(statistics.inverse)];
    let needed: Vec<(ValueId, IntermediateId)> = outputs
        .iter()
        .zip(sources)
        .filter_map(|(&value, source)| Some((value, source?)))
        .filter(|(value, _)| cx.needed.contains(value))
        .collect();
    if needed.is_empty() {
        return Ok(());
    }
    let mut stage = group.stage(cx.named("statistics"), &statistics.shape);
    for (value, source) in needed {
        let statistic = read_whole(&mut stage, source);
        let dtype = cx.graph.value(value).ty().dtype();
        stage.output(Binding::Value(value), convert(statistic, DType::F32, dtype));
    }
    push(cx, group, stage)
}

/// Inference `BatchNormalization`, one elementwise stage:
/// `(x - mean) / sqrt(var + epsilon) * scale + bias`, per channel.
pub(super) fn batch_norm(
    cx: &NodeContext<'_>,
    attrs: BatchNormAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [input, scale, bias, mean, variance] = cx.values()?;
    cx.one_stage(|stage| {
        let x = stage.broadcast(input);
        let channel = channel_index(stage);
        let mut parameter = |value| wide(&stage.load(Binding::Value(value), vec![channel]));
        let (scale, bias, mean, variance) =
            (parameter(scale), parameter(bias), parameter(mean), parameter(variance));
        let deviation = unary(UnaryOp::Sqrt, add(variance, f32(attrs.epsilon.get())));
        let normalized = div(sub(wide(&x), mean), deviation);
        Ok(demote(add(mul(normalized, scale), bias), x.dtype))
    })
}

/// `InstanceNormalization`: statistics per sample and channel over the
/// spatial axes, then the normalised tensor scaled and shifted per channel.
pub(super) fn instance_norm(
    cx: &NodeContext<'_>,
    attrs: InstanceNormAttrs,
) -> Result<LoweredGroup, LowerError> {
    const FIRST_SPATIAL: usize = 2;
    let [input, scale, bias] = cx.values()?;
    let Some(output) = cx.needed_output() else { return Ok(LoweredGroup::default()) };
    let mut group = cx.group();
    let centering = (Centering::Mean, attrs.epsilon);
    let statistics = statistics(cx, &mut group, input, FIRST_SPATIAL, centering)?;
    let mut stage = group.stage(cx.name(), cx.graph.value(input).ty().shape());
    let x = stage.broadcast(input);
    let channel = channel_index(&mut stage);
    let scale = wide(&stage.load(Binding::Value(scale), vec![channel]));
    let bias = wide(&stage.load(Binding::Value(bias), vec![channel]));
    let normalized = normalized(&mut stage, &wide(&x), &statistics, FIRST_SPATIAL);
    stage.output(Binding::Value(output), demote(add(mul(normalized, scale), bias), x.dtype));
    push(cx, &mut group, stage)?;
    Ok(group.finish())
}

/// The channel of an `[N, C, ...]` output element; a 1-D input has one
/// channel, at 0.
fn channel_index(stage: &mut StageBuilder<'_>) -> ExprId {
    match stage.spatial() {
        [_, channel, ..] => *channel,
        _ => stage.constant(0),
    }
}

/// `shape` with the axes in `collapsed` reduced to extent 1: one element
/// per row or block.
fn collapsed(
    cx: &NodeContext<'_>,
    shape: &Shape,
    collapsed: impl std::ops::RangeBounds<usize>,
) -> Result<Shape, LowerError> {
    let dims = shape
        .dims()
        .iter()
        .enumerate()
        .map(|(axis, &dim)| if collapsed.contains(&axis) { Dim::new(1) } else { dim })
        .collect();
    Shape::new(dims).map_err(|source| cx.ill_formed(source.into()))
}

/// The index of the statistic for the stage's output element: its spatial
/// axes, with the axes in `collapsed` read at 0.
fn statistic_index(
    stage: &mut StageBuilder<'_>,
    collapsed: impl std::ops::RangeBounds<usize>,
) -> Vec<ExprId> {
    let zero = stage.constant(0);
    let spatial = stage.spatial().to_vec();
    spatial
        .into_iter()
        .enumerate()
        .map(|(axis, var)| if collapsed.contains(&axis) { zero } else { var })
        .collect()
}

/// An intermediate of the stage's own shape, read at the stage's output
/// element.
fn read_whole(stage: &mut StageBuilder<'_>, intermediate: IntermediateId) -> TensorExpr {
    let spatial = stage.spatial().to_vec();
    stage.load(Binding::Intermediate(intermediate), spatial).expr
}

fn push(
    cx: &NodeContext<'_>,
    group: &mut GroupBuilder<'_>,
    stage: StageBuilder<'_>,
) -> Result<(), LowerError> {
    let stage = cx.finish(stage)?;
    group.push(stage);
    Ok(())
}
