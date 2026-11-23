//! `Slice` and `Split`, resolved to explicit extents.

use super::{input_dims, invalid};
use crate::context::NodeContext;
use crate::error::ImportError;
use ptxc_core::{Dim, Rank};
use ptxc_graph::{Op, SliceAttrs, SliceAxis, SliceStep, SplitAttrs};

/// `Slice(data, starts, ends, axes?, steps?)`, each axis resolved to a
/// start, step and length with numpy's clamping.
pub(super) fn slice(context: &NodeContext<'_>) -> Result<Op, ImportError> {
    let dims = input_dims(context)?;
    let rank = Rank::new(dims.len());
    let starts = context.ints(1)?;
    let ends = context.ints(2)?;
    let axes = if context.has_input(3) {
        context.ints(3)?
    } else {
        (0..starts.len()).map(|axis| i64::try_from(axis).unwrap_or(i64::MAX)).collect()
    };
    let steps = if context.has_input(4) { context.ints(4)? } else { vec![1; starts.len()] };
    if [ends.len(), axes.len(), steps.len()].iter().any(|&length| length != starts.len()) {
        return Err(invalid("Slice's starts, ends, axes and steps differ in length"));
    }
    let mut slices = Vec::with_capacity(starts.len());
    for index in 0..starts.len() {
        let axis = context.axis(axes[index], rank)?;
        let extent =
            i64::try_from(dims[axis.index()].get()).map_err(|_| invalid("an axis is too long"))?;
        slices.push(slice_axis(axis, starts[index], ends[index], steps[index], extent)?);
    }
    Ok(Op::Slice(SliceAttrs { axes: slices }))
}

/// One axis of ONNX `Slice`, as numpy slices, which ONNX's reference
/// implementation uses: negative indices count from the end, then starts and
/// ends are clamped to `[0, extent]` for positive steps and `[-1, extent - 1]`
/// for negative ones.
fn slice_axis(
    axis: ptxc_core::Axis,
    start: i64,
    end: i64,
    step: i64,
    extent: i64,
) -> Result<SliceAxis, ImportError> {
    let step_value = SliceStep::new(step).ok_or_else(|| invalid("a Slice step is zero"))?;
    let wrap = |index: i64| if index < 0 { index.saturating_add(extent) } else { index };
    let (low, high) = if step > 0 { (0, extent) } else { (-1, extent - 1) };
    let start = wrap(start).clamp(low, high);
    let end = wrap(end).clamp(low, high);
    let span = i128::from(end) - i128::from(start);
    let step_wide = i128::from(step);
    let length = if span.signum() == step_wide.signum() {
        (span + step_wide - step_wide.signum()) / step_wide
    } else {
        0
    };
    let length = u64::try_from(length).map_err(|_| invalid("a Slice is too long"))?;
    let start = if length == 0 {
        0
    } else {
        u64::try_from(start).map_err(|_| invalid("a Slice starts before its axis"))?
    };
    Ok(SliceAxis { axis, start, step: step_value, length: Dim::new(length) })
}

/// `Split` with explicit sizes; without them, `num_outputs` (opset 18) or
/// the output count equal pieces, the last smaller if they do not divide.
pub(super) fn split(context: &mut NodeContext<'_>) -> Result<Op, ImportError> {
    let dims = input_dims(context)?;
    let axis = context.axis_attr("axis", 0, Rank::new(dims.len()))?;
    let extent = dims[axis.index()].get();
    let num_outputs = context.attrs.int("num_outputs")?;
    let sizes: Vec<u64> = if context.has_input(1) {
        context
            .ints(1)?
            .iter()
            .map(|&size| u64::try_from(size))
            .collect::<Result<_, _>>()
            .map_err(|_| invalid("a negative Split size"))?
    } else {
        let count = match num_outputs {
            Some(count) => u64::try_from(count).map_err(|_| invalid("num_outputs is negative"))?,
            None => u64::try_from(context.output_count()).unwrap_or(0),
        };
        equal_pieces(extent, count)?
    };
    Ok(Op::Split(SplitAttrs { axis, sizes: sizes.into_iter().map(Dim::new).collect() }))
}

fn equal_pieces(extent: u64, count: u64) -> Result<Vec<u64>, ImportError> {
    if count == 0 {
        return Err(invalid("Split into no pieces"));
    }
    let piece = extent.div_ceil(count);
    let last = extent
        .checked_sub(piece * (count - 1))
        .ok_or_else(|| invalid("Split cannot make that many pieces"))?;
    Ok((0..count).map(|index| if index + 1 == count { last } else { piece }).collect())
}
