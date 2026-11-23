//! `Concat`, `Slice` and `Split`.

use super::{clamped, position, replaced, shifted};
use crate::compose::{less, select};
use crate::error::LowerError;
use crate::ops::NodeContext;
use crate::stage::{Binding, LoweredGroup};
use ptxc_core::{Axis, DType, Dim};
use ptxc_graph::{SliceAttrs, SplitAttrs, ValueId};
use ptxc_loop::{Scalar, TensorExpr};

/// `Concat` along `axis`: the operand whose span holds the output index,
/// chosen by comparing positions. Each operand's read is clamped into it,
/// since the select evaluates every operand.
pub(in crate::ops) fn concat(cx: &NodeContext<'_>, axis: Axis) -> Result<LoweredGroup, LowerError> {
    let (dtype, spans) = spans(cx, axis)?;
    cx.one_stage(|stage| {
        let spatial = stage.spatial().to_vec();
        let here = *spatial.get(axis.index()).ok_or(LowerError::Operands(cx.id))?;
        let mut reads: Vec<(i64, TensorExpr)> = Vec::with_capacity(spans.len());
        for &Span { operand, start, extent } in &spans {
            let local = shifted(stage.exprs(), here, -start);
            let local = clamped(stage.exprs(), local, extent - 1);
            let read = stage.load(Binding::Value(operand), replaced(&spatial, axis, local)).expr;
            reads.push((start + extent, read));
        }
        // Every operand empty: so is the output, and nothing is read.
        let Some((_, last)) = reads.pop() else {
            return Ok(TensorExpr::Const(Scalar::zero(dtype)));
        };
        Ok(reads.into_iter().rev().fold(last, |later, (end, read)| {
            let before_end = less(position(here), TensorExpr::Const(Scalar::i64(end)));
            select(before_end, read, later)
        }))
    })
}

/// Where one operand's elements lie along the concatenated axis.
struct Span {
    operand: ValueId,
    start: i64,
    extent: i64,
}

/// The element type and the spans of the non-empty operands, in order.
fn spans(cx: &NodeContext<'_>, axis: Axis) -> Result<(DType, Vec<Span>), LowerError> {
    let operands = cx.operands();
    let [first, ..] = operands[..] else { return Err(LowerError::Operands(cx.id)) };
    let mut spans = Vec::new();
    let mut start = 0_i64;
    for operand in operands {
        let dim = cx.graph.value(operand).ty().shape().dim(axis);
        let extent = cx.signed(dim.ok_or(LowerError::Operands(cx.id))?)?;
        if extent > 0 {
            spans.push(Span { operand, start, extent });
        }
        start = start.saturating_add(extent);
    }
    Ok((cx.graph.value(first).ty().dtype(), spans))
}

/// `Slice`: along each sliced axis, `start + step * i`.
pub(in crate::ops) fn slice(
    cx: &NodeContext<'_>,
    attrs: &SliceAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    cx.one_stage(|stage| {
        let mut index = stage.spatial().to_vec();
        for sliced in &attrs.axes {
            let slot = index.get_mut(sliced.axis.index()).ok_or(LowerError::Operands(cx.id))?;
            let start = cx.signed(Dim::new(sliced.start))?;
            let exprs = stage.exprs();
            let step = exprs.constant(sliced.step.get());
            let scaled = exprs.mul(*slot, step);
            *slot = shifted(exprs, scaled, start);
        }
        Ok(stage.load(Binding::Value(input), index).expr)
    })
}

/// `Split`: a stage per needed piece, reading the input from the piece's
/// offset along the axis.
pub(in crate::ops) fn split(
    cx: &NodeContext<'_>,
    attrs: &SplitAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    let mut group = cx.group();
    let mut start = 0_i64;
    for (piece, (&output, &size)) in cx.node.outputs().iter().zip(&attrs.sizes).enumerate() {
        let offset = start;
        start = start.saturating_add(cx.signed(size)?);
        if !cx.needed.contains(&output) {
            continue;
        }
        let mut stage =
            group.stage(cx.named(&piece.to_string()), cx.graph.value(output).ty().shape());
        let spatial = stage.spatial().to_vec();
        let here = *spatial.get(attrs.axis.index()).ok_or(LowerError::Operands(cx.id))?;
        let from = shifted(stage.exprs(), here, offset);
        let read = stage.load(Binding::Value(input), replaced(&spatial, attrs.axis, from)).expr;
        stage.output(Binding::Value(output), read);
        let stage = cx.finish(stage)?;
        group.push(stage);
    }
    Ok(group.finish())
}
