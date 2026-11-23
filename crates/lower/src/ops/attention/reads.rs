//! Reading attention's operands at a `(batch, head, position, element)`
//! point in either layout, keys and values after their past, and the bias
//! each score gets from the mask, causality and non-pad lengths.

use super::geometry::Geometry;
use crate::builder::StageBuilder;
use crate::compose::{binary, f32, select};
use crate::error::LowerError;
use crate::ops::NodeContext;
use crate::ops::shape::{clamped, position, shifted};
use crate::precision::convert;
use crate::stage::Binding;
use ptxc_core::{DType, Dim};
use ptxc_graph::{AttentionLayout, AttentionMask, Causality, ValueId};
use ptxc_loop::{BinaryOp, ExprId, Scalar, TensorExpr};

/// A `(batch, head, position, element)` point.
pub(super) type Point = [ExprId; 4];

/// `value` as f32.
fn wide(stage: &mut StageBuilder<'_>, value: ValueId, index: Vec<ExprId>) -> TensorExpr {
    let read = stage.load(Binding::Value(value), index);
    convert(read.expr, read.dtype, DType::F32)
}

/// The index of `point` in a tensor of `layout` with `size` elements per
/// head: 4-D `(batch, head, position, element)`, or packed 3-D
/// `(batch, position, head * size + element)`.
pub(super) fn laid_out(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    layout: AttentionLayout,
    size: Dim,
    [batch, head, at, element]: Point,
) -> Result<Vec<ExprId>, LowerError> {
    Ok(match layout {
        AttentionLayout::Heads => vec![batch, head, at, element],
        AttentionLayout::Packed { .. } => {
            let size = stage.constant(cx.signed(size)?);
            let start = stage.exprs().mul(head, size);
            vec![batch, at, stage.exprs().add(start, element)]
        }
    })
}

/// A query element, as f32.
pub(super) fn query(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    geometry: &Geometry,
    point: Point,
) -> Result<TensorExpr, LowerError> {
    let index = laid_out(cx, stage, geometry.layout, geometry.head_size, point)?;
    Ok(wide(stage, geometry.q, index))
}

/// A key or value element at position `point[2]` of the whole sequence,
/// as f32: the past's first, then the current operand's, each read clamped
/// into its tensor since the select evaluates both.
pub(super) fn key_or_value(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    geometry: &Geometry,
    (current, past, size): (ValueId, Option<ValueId>, Dim),
    point: Point,
) -> Result<TensorExpr, LowerError> {
    let [batch, head, at, element] = point;
    let past_length = cx.signed(geometry.past)?;
    let current_length = cx.signed(geometry.kv_sequence)?;
    let from_current = |stage: &mut StageBuilder<'_>| -> Result<TensorExpr, LowerError> {
        let local = shifted(stage.exprs(), at, -past_length);
        let local = clamped(stage.exprs(), local, current_length - 1);
        let index = laid_out(cx, stage, geometry.layout, size, [batch, head, local, element])?;
        Ok(wide(stage, current, index))
    };
    let Some(past) = past.filter(|_| past_length > 0) else { return from_current(stage) };
    let from_past = {
        let local = clamped(stage.exprs(), at, past_length - 1);
        wide(stage, past, vec![batch, head, local, element])
    };
    if current_length == 0 {
        return Ok(from_past);
    }
    let in_past = binary(BinaryOp::Less, position(at), TensorExpr::Const(Scalar::i64(past_length)));
    Ok(select(in_past, from_past, from_current(stage)?))
}

const fn negative_infinity() -> TensorExpr {
    TensorExpr::Const(Scalar::lowest(DType::F32))
}

/// What score `[batch, head, query, key]` gets added before the softmax:
/// negative infinity where causality, a non-pad length or a mask hides the
/// key, the additive mask's value, or 0; `None` when nothing is hidden.
pub(super) fn bias(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    geometry: &Geometry,
    [batch, head, query, key]: Point,
) -> Result<Option<TensorExpr>, LowerError> {
    let mut hidden = Vec::new();
    if geometry.causality == Causality::Causal {
        let last_visible = shifted(stage.exprs(), query, cx.signed(geometry.past)?);
        hidden.push(binary(BinaryOp::Greater, position(key), position(last_visible)));
    }
    if let Some(lengths) = geometry.lengths {
        let length = stage.load(Binding::Value(lengths), vec![batch]).expr;
        hidden.push(binary(BinaryOp::GreaterOrEqual, position(key), length));
    }
    let masked = match geometry.mask {
        Some(mask) => Some(mask_value(cx, stage, geometry, mask, [batch, head, query, key])?),
        None => None,
    };
    let hidden = hidden.into_iter().reduce(|a, b| binary(BinaryOp::Or, a, b));
    Ok(match (hidden, masked) {
        (None, masked) => masked,
        (Some(hidden), masked) => {
            Some(select(hidden, negative_infinity(), masked.unwrap_or_else(|| f32(0.0))))
        }
    })
}

/// The mask's value at a score: broadcast from the right over
/// `(batch, head, query, key)`, its last axis possibly shorter than the
/// keys, past which every key is hidden.
fn mask_value(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    geometry: &Geometry,
    mask: ValueId,
    point: Point,
) -> Result<TensorExpr, LowerError> {
    let dims = cx.graph.value(mask).ty().shape().dims().to_vec();
    let Some((&length, leading)) = dims.split_last() else { return Ok(f32(0.0)) };
    let length = cx.signed(length)?;
    if length == 0 {
        return Ok(negative_infinity());
    }
    let zero = stage.constant(0);
    let key = point[3];
    let leading_points = &point[..3];
    let mut index: Vec<ExprId> = leading
        .iter()
        .rev()
        .zip(leading_points.iter().rev())
        .map(|(dim, &axis)| if dim.get() == 1 { zero } else { axis })
        .collect();
    index.reverse();
    index.push(clamped(stage.exprs(), key, length - 1));
    let read = stage.load(Binding::Value(mask), index);
    let value = match geometry.mask_kind {
        AttentionMask::Boolean => select(read.expr, f32(0.0), negative_infinity()),
        AttentionMask::Additive | AttentionMask::None => convert(read.expr, read.dtype, DType::F32),
    };
    let within = binary(BinaryOp::Less, position(key), TensorExpr::Const(Scalar::i64(length)));
    Ok(select(within, value, negative_infinity()))
}
