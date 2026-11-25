//! Max, average and global pooling: a reduction over each window's taps,
//! the input read through its padding.

use super::NodeContext;
use super::reduce::{load_reduced, wide};
use super::window::{Counted, Span, read, tap_count, window_load};
use crate::builder::{Operand, StageBuilder};
use crate::compose::{div, f32};
use crate::error::LowerError;
use crate::precision::{convert, demote, promote};
use crate::stage::LoweredGroup;
use ptxc_core::{Axis, DType};
use ptxc_graph::{GlobalPoolKind, KeepDims, PadCounting, PoolAttrs, PoolKind, ValueId};
use ptxc_loop::{ExprId, IterVarId, ReduceOp, Scalar, TensorExpr};

/// `MaxPool` or `AveragePool` on `[N, C, D...]`. Max ignores padding;
/// average divides by the taps its counting includes.
pub(super) fn pool(cx: &NodeContext<'_>, attrs: &PoolAttrs) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    let Some(output) = cx.needed_output() else { return Ok(LoweredGroup::default()) };
    let input_dims = cx.graph.value(input).ty().shape().dims().to_vec();
    let output_dims = cx.graph.value(output).ty().shape().dims().to_vec();
    let spans: Vec<Span> = attrs
        .axes
        .iter()
        .zip(input_dims.iter().skip(2).zip(output_dims.iter().skip(2)))
        .map(|(axis, (&input, &output))| Span {
            input,
            output,
            kernel: axis.kernel,
            axis: axis.window,
        })
        .collect();
    let dtype = cx.graph.value(input).ty().dtype();
    cx.one_stage(|stage| {
        let spatial = stage.spatial().to_vec();
        let [n, c, outs @ ..] = &spatial[..] else { return Err(LowerError::Operands(cx.id)) };
        let empty = spans.iter().any(|span| span.input.get() == 0);
        match attrs.kind {
            PoolKind::Max if empty => Ok(TensorExpr::Const(Scalar::lowest(dtype))),
            PoolKind::Max => {
                let padding = Scalar::lowest(dtype);
                let window = load_window(cx, stage, (input, padding), (*n, *c, outs), &spans)?;
                Ok(reduce(ReduceOp::Max, window.taps, window.value.expr))
            }
            PoolKind::Average(counting) => {
                let counted = match counting {
                    PadCounting::Exclude => Counted::Input,
                    PadCounting::Include => Counted::Padded,
                };
                let count = window_count(cx, stage, outs, &spans, counted)?;
                let sum = if empty {
                    f32(0.0)
                } else {
                    let padding = Scalar::zero(dtype);
                    let window = load_window(cx, stage, (input, padding), (*n, *c, outs), &spans)?;
                    reduce(
                        ReduceOp::Sum,
                        window.taps,
                        promote(window.value.expr, window.value.dtype),
                    )
                };
                Ok(demote(div(sum, count), dtype))
            }
        }
    })
}

fn reduce(op: ReduceOp, axes: Vec<IterVarId>, body: TensorExpr) -> TensorExpr {
    TensorExpr::Reduce { op, axes, body: Box::new(body) }
}

/// One window's read of the input: the value at each tap, and the tap
/// axes.
struct Window {
    value: Operand,
    taps: Vec<IterVarId>,
}

/// The input at each tap of output `[n, c, outs...]`'s window, `padding`
/// where a tap reads padding, a reduction axis per spatial axis.
fn load_window(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    (input, padding): (ValueId, Scalar),
    (n, c, outs): (ExprId, ExprId, &[ExprId]),
    spans: &[Span],
) -> Result<Window, LowerError> {
    let (mut index, mut reads, mut taps) = (vec![n, c], Vec::new(), Vec::new());
    for (&out, &span) in outs.iter().zip(spans) {
        let (tap_axis, tap) = stage.reduce_axis(span.kernel);
        let tap_read = read(cx, stage.exprs(), span, (out, tap))?;
        index.push(tap_read.index);
        reads.push(tap_read);
        taps.push(tap_axis);
    }
    Ok(Window { value: window_load(stage, (input, index), &reads, padding), taps })
}

/// How many taps of output `outs`'s window are counted, as an f32 value.
fn window_count(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    outs: &[ExprId],
    spans: &[Span],
    counted: Counted,
) -> Result<TensorExpr, LowerError> {
    let mut total: Option<ExprId> = None;
    for (&out, &span) in outs.iter().zip(spans) {
        let count = tap_count(cx, stage.exprs(), span, out, counted)?;
        total = Some(total.map_or(count, |total| stage.exprs().mul(total, count)));
    }
    let total = total.unwrap_or_else(|| stage.constant(1));
    let count = TensorExpr::Index { index: total, dtype: DType::I64 };
    Ok(convert(count, DType::I64, DType::F32))
}

/// `GlobalMaxPool` or `GlobalAveragePool`: each `[n, c]` plane reduced over
/// every spatial axis.
pub(super) fn global_pool(
    cx: &NodeContext<'_>,
    kind: GlobalPoolKind,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    let rank = cx.graph.value(input).ty().shape().dims().len();
    let axes: Vec<Axis> = (2..rank).map(Axis::new).collect();
    cx.one_stage(|stage| {
        let (x, over) = load_reduced(cx, stage, input, &axes, KeepDims::Keep)?;
        let total = match kind {
            GlobalPoolKind::Max => over.apply(ReduceOp::Max, wide(&x)),
            GlobalPoolKind::Average => div(over.sum(wide(&x)), f32(over.count.to_f32())),
        };
        Ok(demote(total, x.dtype))
    })
}
