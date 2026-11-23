//! `Resize`: each axis reads its input directly when every output position
//! reads one input element, and otherwise interpolates over taps whose
//! positions and weights are tables computed at lowering time (D47).

mod taps;

use super::NodeContext;
use crate::builder::{GroupBuilder, StageBuilder};
use crate::compose::{binary, f32, max, min, mul, select};
use crate::error::LowerError;
use crate::precision::{convert, demote};
use crate::stage::{Binding, LoweredGroup, TableId};
use ptxc_core::convert::f64_to_f32;
use ptxc_core::{DType, Dim, HostTensor};
use ptxc_graph::{CoordinateTransform, ResizeAttrs, ValueId};
use ptxc_loop::{BinaryOp, ExprId, IterVarId, ReduceOp, Scalar, TensorExpr};
use taps::{Axis, Taps};

/// How the stage reads one axis.
enum AxisRead {
    /// Output position `i` reads input position `i`.
    Identity,
    /// Output position `i` reads the input position table `index` holds
    /// at `i`.
    Selected { index: TableId },
    /// Output position `i` reads `width` taps: tap `t` at the position table
    /// `index` holds at `i * width + t`, weighted by table `weight` there.
    Interpolated { index: TableId, weight: TableId, width: usize },
}

/// One axis's read, and the table flagging the output positions that
/// extrapolate, if any do.
struct AxisPlan {
    read: AxisRead,
    extrapolated: Option<TableId>,
}

/// The weight of a tap that is the whole value: nearest modes, and
/// interpolation at an integral coordinate, give exactly this.
const WHOLE: f64 = 1.0;

/// The one position `taps` reads with weight 1, every other tap weighing
/// 0; `None` when it interpolates.
fn selection(taps: &[(i64, f64)]) -> Option<i64> {
    let mut chosen = None;
    for &(position, weight) in taps {
        if weight == 0.0 {
            continue;
        }
        if weight != WHOLE || chosen.is_some() {
            return None;
        }
        chosen = Some(position);
    }
    chosen
}

/// The plan for an axis with these taps, its tables added to `group`. An
/// extrapolating position reads position 0, which the select discards.
fn plan(group: &mut GroupBuilder<'_>, taps: &[Taps]) -> AxisPlan {
    let extrapolated = taps.iter().any(Option::is_none).then(|| {
        let flags: Vec<bool> = taps.iter().map(Option::is_none).collect();
        group.table(HostTensor::vector(flags.into()))
    });
    let selected: Option<Vec<i64>> =
        taps.iter().map(|taps| taps.as_ref().map_or(Some(0), |taps| selection(taps))).collect();
    let read = match selected {
        Some(positions)
            if extrapolated.is_none() && positions.iter().zip(0..).all(|(&p, i)| p == i) =>
        {
            AxisRead::Identity
        }
        Some(positions) => {
            AxisRead::Selected { index: group.table(HostTensor::vector(positions.into())) }
        }
        None => interpolated(group, taps),
    };
    AxisPlan { read, extrapolated }
}

/// Flat index and weight tables, `width` taps per output position.
fn interpolated(group: &mut GroupBuilder<'_>, taps: &[Taps]) -> AxisRead {
    let width = taps.iter().flatten().map(Vec::len).max().unwrap_or(1);
    let (mut positions, mut weights) = (Vec::new(), Vec::new());
    for row in taps {
        let row = row.as_deref().unwrap_or_default();
        for slot in 0..width {
            let (position, weight) = row.get(slot).copied().unwrap_or((0, 0.0));
            positions.push(position);
            weights.push(f64_to_f32(weight));
        }
    }
    AxisRead::Interpolated {
        index: group.table(HostTensor::vector(positions.into())),
        weight: group.table(HostTensor::vector(weights.into())),
        width,
    }
}

/// What the stage reads along every axis for one output element.
struct Reads {
    index: Vec<ExprId>,
    weights: Vec<TensorExpr>,
    taps: Vec<IterVarId>,
    extrapolated: Vec<TensorExpr>,
}

fn reads(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    plans: &[AxisPlan],
) -> Result<Reads, LowerError> {
    let spatial = stage.spatial().to_vec();
    let mut reads = Reads {
        index: Vec::new(),
        weights: Vec::new(),
        taps: Vec::new(),
        extrapolated: Vec::new(),
    };
    for (axis, (plan, &out)) in plans.iter().zip(&spatial).enumerate() {
        let name = format!("j{axis}");
        match plan.read {
            AxisRead::Identity => reads.index.push(out),
            AxisRead::Selected { index } => {
                reads.index.push(stage.loaded(&name, Binding::Table(index), vec![out]));
            }
            AxisRead::Interpolated { index, weight, width } => {
                let width_dim = Dim::new(u64::try_from(width).unwrap_or(u64::MAX));
                let (tap_axis, tap) = stage.reduce_axis(width_dim);
                let stride = stage.constant(cx.signed(width_dim)?);
                let row = stage.exprs().mul(out, stride);
                let slot = stage.exprs().add(row, tap);
                reads.index.push(stage.loaded(&name, Binding::Table(index), vec![slot]));
                reads.weights.push(stage.load(Binding::Table(weight), vec![slot]).expr);
                reads.taps.push(tap_axis);
            }
        }
        if let Some(flags) = plan.extrapolated {
            reads.extrapolated.push(stage.load(Binding::Table(flags), vec![out]).expr);
        }
    }
    Ok(reads)
}

/// `Resize`: every output element is the tensor product of its axes' taps,
/// or the extrapolation value where any axis extrapolates. Integers
/// interpolate in f32 and truncate and saturate.
pub(super) fn resize(
    cx: &NodeContext<'_>,
    attrs: &ResizeAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    let Some(output) = cx.needed_output() else { return Ok(LoweredGroup::default()) };
    let input_dims = cx.graph.value(input).ty().shape().dims().to_vec();
    let output_dims = cx.graph.value(output).ty().shape().dims().to_vec();
    let dtype = cx.graph.value(input).ty().dtype();
    let fill = extrapolation(attrs, dtype);
    let mut group = cx.group();
    let shape = cx.graph.value(output).ty().shape();
    // An empty input has nothing to read; every output extrapolates.
    let result = if input_dims.iter().any(|dim| dim.get() == 0) {
        let mut stage = group.stage(cx.name(), shape);
        stage.output(Binding::Value(output), fill);
        stage
    } else {
        let axes = resized_axes(cx, attrs, &input_dims, &output_dims)?;
        let plans: Vec<AxisPlan> =
            axes.iter().map(|axis| plan(&mut group, &axis.taps(attrs))).collect();
        let mut stage = group.stage(cx.name(), shape);
        let value = resampled(cx, &mut stage, input, &plans, fill)?;
        stage.output(Binding::Value(output), value);
        stage
    };
    let stage = cx.finish(result)?;
    group.push(stage);
    Ok(group.finish())
}

/// One output element: the input read along every axis, weighted and
/// summed over the interpolating axes' taps, or `fill` where any axis
/// extrapolates.
fn resampled(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    input: ValueId,
    plans: &[AxisPlan],
    fill: TensorExpr,
) -> Result<TensorExpr, LowerError> {
    let reads = reads(cx, stage, plans)?;
    let x = stage.load(Binding::Value(input), reads.index);
    let value = match reads.weights.into_iter().reduce(mul) {
        None => x.expr,
        Some(weight) => {
            let body = mul(weight, convert(x.expr, x.dtype, DType::F32));
            let sum =
                TensorExpr::Reduce { op: ReduceOp::Sum, axes: reads.taps, body: Box::new(body) };
            from_f32(sum, x.dtype)
        }
    };
    Ok(match reads.extrapolated.into_iter().reduce(|a, b| binary(BinaryOp::Or, a, b)) {
        Some(extrapolates) => select(extrapolates, fill, value),
        None => value,
    })
}

/// The axes as tap computations need them.
fn resized_axes(
    cx: &NodeContext<'_>,
    attrs: &ResizeAttrs,
    input: &[Dim],
    output: &[Dim],
) -> Result<Vec<Axis>, LowerError> {
    let regions: Vec<Option<[f64; 2]>> = match &attrs.coordinates {
        CoordinateTransform::TfCropAndResize { roi, .. } => roi
            .iter()
            .map(|[start, end]| Some([f64::from(start.get()), f64::from(end.get())]))
            .collect(),
        _ => vec![None; attrs.axes.len()],
    };
    attrs
        .axes
        .iter()
        .zip(regions)
        .zip(input.iter().zip(output))
        .map(|((axis, roi), (&input, &output))| {
            Ok(Axis {
                input: cx.signed(input)?,
                output: cx.signed(output)?,
                scale: f64::from(axis.scale.get()),
                roi,
            })
        })
        .collect()
}

/// An f32 result in `dtype`: rounded for floats; for integers truncated
/// and saturated, as the reference treats its f64 result.
fn from_f32(value: TensorExpr, dtype: DType) -> TensorExpr {
    if dtype.is_float() {
        return demote(value, dtype);
    }
    // NOTE: the reference interpolates integers in f64; ptxc uses f32, so a
    // truncated result can land one step from the reference's.
    let truncated = convert(value, DType::F32, DType::I64);
    let (lowest, highest) = (Scalar::lowest(dtype), Scalar::highest(dtype));
    let widen = |bound: Scalar| convert(TensorExpr::Const(bound), dtype, DType::I64);
    let saturated = min(max(truncated, widen(lowest)), widen(highest));
    convert(saturated, DType::I64, dtype)
}

/// The value where `tf_crop_and_resize` extrapolates, in `dtype`; 0 for
/// every other transform, which reaches it only on an empty input.
fn extrapolation(attrs: &ResizeAttrs, dtype: DType) -> TensorExpr {
    let value = match &attrs.coordinates {
        CoordinateTransform::TfCropAndResize { extrapolation, .. } => extrapolation.get(),
        _ => 0.0,
    };
    from_f32(f32(value), dtype)
}
