//! Grouped N-d convolution: one reduction over a group's input channels
//! and every kernel tap, masked where a tap reads padding, with the bias
//! added after it.

use super::NodeContext;
use super::window::{Span, masked, read};
use crate::compose::{add, f32, mul};
use crate::error::LowerError;
use crate::precision::{demote, promote};
use crate::stage::{Binding, LoweredGroup, StageBuilder};
use ptxc_core::Dim;
use ptxc_graph::{ConvAttrs, ValueId};
use ptxc_loop::{ExprArena, ExprId, ReduceOp, TensorExpr};

/// The shape of a convolution: each spatial axis's window, the input
/// channels per group, and the output channels per group.
struct Geometry {
    spans: Vec<Span>,
    group_channels: Dim,
    per_group: Dim,
}

/// `Conv(x, w, b?)`: `x: [N, C, D...]` by `w: [M, C / groups, K...]` into
/// `[N, M, O...]`, each output channel reading its group's input channels.
pub(super) fn conv(cx: &NodeContext<'_>, attrs: &ConvAttrs) -> Result<LoweredGroup, LowerError> {
    let (input, weights, bias) = match cx.operands()[..] {
        [input, weights] => (input, weights, None),
        [input, weights, bias] => (input, weights, Some(bias)),
        _ => return Err(LowerError::Operands(cx.id)),
    };
    let Some(output) = cx.needed_output() else { return Ok(LoweredGroup::default()) };
    let geometry = geometry(cx, attrs, [input, weights, output])?;
    let dtype = cx.graph.value(input).ty().dtype();
    cx.one_stage(|stage| {
        let spatial = stage.spatial().to_vec();
        let [sample, feature, outs @ ..] = &spatial[..] else {
            return Err(LowerError::Operands(cx.id));
        };
        let sum = convolved(cx, stage, [input, weights], &geometry, (*sample, *feature, outs))?;
        let biased = match bias {
            Some(bias) => {
                let shift = stage.load(Binding::Value(bias), vec![*feature]);
                add(sum, promote(shift.expr, shift.dtype))
            }
            None => sum,
        };
        Ok(demote(biased, dtype))
    })
}

fn geometry(
    cx: &NodeContext<'_>,
    attrs: &ConvAttrs,
    [input, weights, output]: [ValueId; 3],
) -> Result<Geometry, LowerError> {
    let dims = |value: ValueId| cx.graph.value(value).ty().shape().dims().to_vec();
    let (input_dims, weight_dims, output_dims) = (dims(input), dims(weights), dims(output));
    let [out_channels, group_channels, kernel @ ..] = &weight_dims[..] else {
        return Err(LowerError::Operands(cx.id));
    };
    let spans = attrs
        .axes
        .iter()
        .zip(kernel)
        .zip(input_dims.iter().skip(2).zip(output_dims.iter().skip(2)))
        .map(|((&axis, &kernel), (&input, &output))| Span { input, output, kernel, axis })
        .collect();
    let per_group = Dim::new(out_channels.get() / attrs.groups.get());
    Ok(Geometry { spans, group_channels: *group_channels, per_group })
}

/// The sum, over output `[sample, feature, outs...]`'s group channels and
/// kernel taps, of input times weight; 0 when the input has nothing to
/// read.
fn convolved(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    [input, weights]: [ValueId; 2],
    geometry: &Geometry,
    (sample, feature, outs): (ExprId, ExprId, &[ExprId]),
) -> Result<TensorExpr, LowerError> {
    let empty = |dim: Dim| dim.get() == 0;
    if empty(geometry.group_channels) || geometry.spans.iter().any(|span| empty(span.input)) {
        return Ok(f32(0.0));
    }
    let (channel_axis, channel) = stage.reduce_axis(geometry.group_channels);
    let read_channel = input_channel(cx, stage.exprs(), (feature, channel), geometry)?;
    let (mut input_index, mut weight_index) = (vec![sample, read_channel], vec![feature, channel]);
    let (mut axes, mut inside) = (vec![channel_axis], Vec::new());
    for (&out, &span) in outs.iter().zip(&geometry.spans) {
        let (tap_axis, tap) = stage.reduce_axis(span.kernel);
        let tap_read = read(cx, stage.exprs(), span, (out, tap))?;
        input_index.push(tap_read.index);
        weight_index.push(tap);
        inside.extend(tap_read.inside);
        axes.push(tap_axis);
    }
    let value = stage.load(Binding::Value(input), input_index);
    let weight = stage.load(Binding::Value(weights), weight_index);
    let product = mul(promote(value.expr, value.dtype), promote(weight.expr, weight.dtype));
    let body = masked(inside, product, f32(0.0));
    Ok(TensorExpr::Reduce { op: ReduceOp::Sum, axes, body: Box::new(body) })
}

/// The input channel that channel `channel` of output channel `feature`'s
/// group reads: `(feature / per_group) * group_channels + channel`.
fn input_channel(
    cx: &NodeContext<'_>,
    exprs: &mut ExprArena,
    (feature, channel): (ExprId, ExprId),
    geometry: &Geometry,
) -> Result<ExprId, LowerError> {
    let per_group = exprs.constant(cx.signed(geometry.per_group)?);
    let group = exprs.floor_div(feature, per_group);
    let width = exprs.constant(cx.signed(geometry.group_channels)?);
    let first = exprs.mul(group, width);
    Ok(exprs.add(first, channel))
}
