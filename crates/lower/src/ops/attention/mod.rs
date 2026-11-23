//! ONNX `Attention`, unfused (D48): the scores `softcap(scale * Q K^T) + bias`,
//! each row's maximum and sum of exponentials, then `Y = softmax(scores) V`,
//! each a stage over f32 intermediates. Present keys and values and the QK
//! intermediate are stages of their own, built only when read.

mod geometry;
mod reads;

use super::NodeContext;
use crate::builder::{GroupBuilder, StageBuilder};
use crate::compose::{add, div, f32, mul, sub, unary};
use crate::error::LowerError;
use crate::precision::convert;
use crate::stage::{Binding, IntermediateId, LoweredGroup};
use geometry::Geometry;
use ptxc_core::{DType, Dim, Shape};
use ptxc_graph::{AttentionAttrs, F32Attr, Presence, QkOutput, ValueId};
use ptxc_loop::{ReduceOp, TensorExpr, UnaryOp};
use reads::{Point, bias, key_or_value, query};

/// `dims` as a shape; past `u64` the error `ComputeBuilder` would give.
fn shape(cx: &NodeContext<'_>, dims: &[Dim]) -> Result<Shape, LowerError> {
    Shape::new(dims.to_vec()).map_err(|source| cx.ill_formed(source.into()))
}

/// The `(batch, head, query, key)` spatial point of a stage over scores.
fn score_point(cx: &NodeContext<'_>, stage: &StageBuilder<'_>) -> Result<Point, LowerError> {
    match *stage.spatial() {
        [batch, head, at, key] => Ok([batch, head, at, key]),
        _ => Err(LowerError::Operands(cx.id)),
    }
}

/// `scale * sum over d of q * k` for score `[batch, head, query, key]`,
/// softcapped if `capped` and a cap is set.
fn scaled_scores(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    (geometry, attrs): (&Geometry, &AttentionAttrs),
    [batch, head, at, key]: Point,
    capped: bool,
) -> Result<TensorExpr, LowerError> {
    let (d_axis, d) = stage.reduce_axis(geometry.head_size);
    let kv_head = geometry.kv_head(cx, stage, head)?;
    let q = query(cx, stage, geometry, [batch, head, at, d])?;
    let keys = (geometry.k, geometry.past_key, geometry.head_size);
    let k = key_or_value(cx, stage, geometry, keys, [batch, kv_head, key, d])?;
    let dot =
        TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![d_axis], body: Box::new(mul(q, k)) };
    let scaled = mul(f32(attrs.scale.get()), dot);
    let cap = attrs.softcap.map(F32Attr::get).filter(|&cap| capped && cap > 0.0);
    Ok(match cap {
        Some(cap) => mul(f32(cap), unary(UnaryOp::Tanh, div(scaled, f32(cap)))),
        None => scaled,
    })
}

/// The scores and each row's maximum and sum of exponentials.
struct Softmax {
    scores: IntermediateId,
    max: IntermediateId,
    sum: IntermediateId,
}

fn softmax(
    cx: &NodeContext<'_>,
    group: &mut GroupBuilder<'_>,
    geometry: &Geometry,
    attrs: &AttentionAttrs,
) -> Result<Softmax, LowerError> {
    let score_shape = shape(cx, &geometry.score_dims())?;
    let scores = group.intermediate(&score_shape);
    let mut stage = group.stage(cx.named("scores"), &score_shape);
    let point = score_point(cx, &stage)?;
    let scaled = scaled_scores(cx, &mut stage, (geometry, attrs), point, true)?;
    let biased = match bias(cx, &mut stage, geometry, point)? {
        Some(bias) => add(scaled, bias),
        None => scaled,
    };
    stage.output(Binding::Intermediate(scores), biased);
    let stage = cx.finish(stage)?;
    group.push(stage);
    let max = row_reduction(cx, group, geometry, (scores, None))?;
    let sum = row_reduction(cx, group, geometry, (scores, Some(max)))?;
    Ok(Softmax { scores, max, sum })
}

/// A stage reducing each row of `scores`: its maximum, or with the row's
/// maximum given, its sum of `exp(score - max)`.
fn row_reduction(
    cx: &NodeContext<'_>,
    group: &mut GroupBuilder<'_>,
    geometry: &Geometry,
    (scores, max): (IntermediateId, Option<IntermediateId>),
) -> Result<IntermediateId, LowerError> {
    let row_shape = shape(cx, &geometry.row_dims())?;
    let row = group.intermediate(&row_shape);
    let name = if max.is_some() { "sum" } else { "max" };
    let mut stage = group.stage(cx.named(name), &row_shape);
    let [batch, head, at, zero] = score_point(cx, &stage)?;
    let (key_axis, key) = stage.reduce_axis(geometry.total());
    let score = stage.load(Binding::Intermediate(scores), vec![batch, head, at, key]).expr;
    let (op, body) = match max {
        None => (ReduceOp::Max, score),
        Some(max) => {
            let max = stage.load(Binding::Intermediate(max), vec![batch, head, at, zero]).expr;
            (ReduceOp::Sum, unary(UnaryOp::Exp, sub(score, max)))
        }
    };
    stage.output(
        Binding::Intermediate(row),
        TensorExpr::Reduce { op, axes: vec![key_axis], body: Box::new(body) },
    );
    let stage = cx.finish(stage)?;
    group.push(stage);
    Ok(row)
}

/// `exp(score - max) / sum` at score `[batch, head, at, key]`.
fn probability(
    stage: &mut StageBuilder<'_>,
    softmax: &Softmax,
    [batch, head, at, key]: Point,
) -> TensorExpr {
    let zero = stage.constant(0);
    let score = stage.load(Binding::Intermediate(softmax.scores), vec![batch, head, at, key]).expr;
    let max = stage.load(Binding::Intermediate(softmax.max), vec![batch, head, at, zero]).expr;
    let sum = stage.load(Binding::Intermediate(softmax.sum), vec![batch, head, at, zero]).expr;
    div(unary(UnaryOp::Exp, sub(score, max)), sum)
}

/// `Y = sum over keys of probability * value`, laid out as `Y` is.
fn output_stage(
    cx: &NodeContext<'_>,
    group: &mut GroupBuilder<'_>,
    geometry: &Geometry,
    (softmax, y): (&Softmax, ValueId),
) -> Result<(), LowerError> {
    let ty = cx.graph.value(y).ty();
    let mut stage = group.stage(cx.name(), ty.shape());
    let [batch, head, at, element] = geometry.y_point(cx, &mut stage)?;
    let (key_axis, key) = stage.reduce_axis(geometry.total());
    let kv_head = geometry.kv_head(cx, &mut stage, head)?;
    let weight = probability(&mut stage, softmax, [batch, head, at, key]);
    let values = (geometry.v, geometry.past_value, geometry.v_head_size);
    let value = key_or_value(cx, &mut stage, geometry, values, [batch, kv_head, key, element])?;
    let sum = TensorExpr::Reduce {
        op: ReduceOp::Sum,
        axes: vec![key_axis],
        body: Box::new(mul(weight, value)),
    };
    stage.output(Binding::Value(y), convert(sum, DType::F32, ty.dtype()));
    let stage = cx.finish(stage)?;
    group.push(stage);
    Ok(())
}

/// The past and current keys or values, `(batch, kv_head, total, size)`.
fn present_stage(
    cx: &NodeContext<'_>,
    group: &mut GroupBuilder<'_>,
    geometry: &Geometry,
    (output, source): (ValueId, (ValueId, Option<ValueId>, Dim)),
) -> Result<(), LowerError> {
    let ty = cx.graph.value(output).ty();
    let mut stage = group.stage(cx.named("present"), ty.shape());
    let point = score_point(cx, &stage)?;
    let read = key_or_value(cx, &mut stage, geometry, source, point)?;
    stage.output(Binding::Value(output), convert(read, DType::F32, ty.dtype()));
    let stage = cx.finish(stage)?;
    group.push(stage);
    Ok(())
}

/// The QK intermediate at the stage `kind` names.
fn qk_stage(
    cx: &NodeContext<'_>,
    group: &mut GroupBuilder<'_>,
    (geometry, attrs): (&Geometry, &AttentionAttrs),
    (output, kind, softmax): (ValueId, QkOutput, &Softmax),
) -> Result<(), LowerError> {
    let ty = cx.graph.value(output).ty();
    let mut stage = group.stage(cx.named("qk"), ty.shape());
    let point = score_point(cx, &stage)?;
    let scores = match kind {
        QkOutput::MatMul => scaled_scores(cx, &mut stage, (geometry, attrs), point, false)?,
        QkOutput::Softcapped => scaled_scores(cx, &mut stage, (geometry, attrs), point, true)?,
        QkOutput::Masked => stage.load(Binding::Intermediate(softmax.scores), point.to_vec()).expr,
        QkOutput::Softmax => probability(&mut stage, softmax, point),
    };
    stage.output(Binding::Value(output), convert(scores, DType::F32, ty.dtype()));
    let stage = cx.finish(stage)?;
    group.push(stage);
    Ok(())
}

/// `Attention`'s outputs: `Y`, then the present key and value if asked
/// for, then the QK intermediate if asked for; each built only if read.
pub(in crate::ops) fn attention(
    cx: &NodeContext<'_>,
    attrs: &AttentionAttrs,
) -> Result<LoweredGroup, LowerError> {
    let geometry = Geometry::of(cx, attrs)?;
    let outputs: Vec<ValueId> = cx.node.outputs().iter().copied().collect();
    let needed = |value: &&ValueId| cx.needed.contains(value);
    let mut group = cx.group();
    let softmax = softmax(cx, &mut group, &geometry, attrs)?;
    let mut rest = outputs.iter();
    if let Some(&y) = rest.next().filter(needed) {
        output_stage(cx, &mut group, &geometry, (&softmax, y))?;
    }
    if attrs.present == Presence::Present {
        let keys = (geometry.k, geometry.past_key, geometry.head_size);
        let values = (geometry.v, geometry.past_value, geometry.v_head_size);
        for (output, source) in rest.by_ref().take(2).zip([keys, values]) {
            if needed(&output) {
                present_stage(cx, &mut group, &geometry, (*output, source))?;
            }
        }
    }
    if let (Some(kind), Some(&qk)) = (attrs.qk_output, rest.next().filter(needed)) {
        qk_stage(cx, &mut group, (&geometry, attrs), (qk, kind, &softmax))?;
    }
    Ok(group.finish())
}
