//! `Pad`, with its widths and value resolved, and `Trilu`.

use super::invalid;
use crate::context::NodeContext;
use crate::error::ImportError;
use crate::lowered::{Lowered, Operand};
use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16, Overflow};
use ptxc_core::{Axis, DType, Dim, HostTensor, Rank, Shape, TensorData};
use ptxc_graph::{
    AxisPadding, DiagonalOffset, Op, PadAttrs, PadMode, Padding, SliceAttrs, SliceAxis, SliceStep,
    TriangularPart, TriluAttrs,
};

/// One axis's ONNX pad widths; a negative width removes elements.
#[derive(Clone, Copy, Default)]
struct SignedWidths {
    begin: i64,
    end: i64,
}

/// `Pad`: `pads` lists begins then ends for every axis, or for the `axes`
/// input (opset 18). The constant value defaults to zero. Negative widths
/// crop: they become a `Slice` before the `Pad` (D32), or replace it when
/// nothing is added.
pub(super) fn pad(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let mode = pad_mode(context)?;
    let ty = context.ty(0)?.clone();
    let widths = signed_widths(context, ty.shape().rank())?;
    let crop = crop(&widths, ty.shape().dims())?;
    // NOTE: ONNX leaves the order of cropping and padding undefined (its
    // reference passes negative widths to np.pad, which rejects them). For
    // constant and edge padding the order cannot change the result; for
    // reflect and wrap it can, so ptxc does not guess.
    if !crop.axes.is_empty() && matches!(mode, PadMode::Reflect | PadMode::Wrap) {
        return Err(context.attrs.unsupported("negative pads"));
    }
    let growth = PadAttrs { mode, axes: widths.iter().map(|&widths| added(widths)).collect() };
    let grows = growth.axes.iter().any(|axis| axis.begin.get() > 0 || axis.end.get() > 0);
    let rest =
        if mode == PadMode::Constant { vec![pad_value(context, ty.dtype())?] } else { vec![] };
    Ok(match (crop.axes.is_empty(), grows) {
        (true, _) => {
            let operands = std::iter::once(Operand::Input(0)).chain(rest).collect();
            Lowered::Node { op: Op::Pad(growth), operands }
        }
        (false, false) => Lowered::on_inputs(Op::Slice(crop), 1),
        (false, true) => Lowered::Chain { first: Op::Slice(crop), then: Op::Pad(growth), rest },
    })
}

fn pad_mode(context: &mut NodeContext<'_>) -> Result<PadMode, ImportError> {
    Ok(match context.attrs.string_or("mode", "constant")?.as_str() {
        "constant" => PadMode::Constant,
        "reflect" => PadMode::Reflect,
        "edge" => PadMode::Edge,
        "wrap" => PadMode::Wrap,
        _ => return Err(context.attrs.unsupported("mode")),
    })
}

/// Every axis's widths, zero for axes `axes` leaves out.
fn signed_widths(context: &NodeContext<'_>, rank: Rank) -> Result<Vec<SignedWidths>, ImportError> {
    let pads = context.ints(1)?;
    let axes: Vec<usize> = if context.has_input(3) {
        context
            .ints(3)?
            .iter()
            .map(|&axis| context.axis(axis, rank).map(ptxc_core::Axis::index))
            .collect::<Result<_, _>>()?
    } else {
        (0..rank.get()).collect()
    };
    if pads.len() != 2 * axes.len() {
        return Err(invalid("Pad needs a begin and an end for each axis"));
    }
    let mut widths = vec![SignedWidths::default(); rank.get()];
    for (index, &axis) in axes.iter().enumerate() {
        widths[axis] = SignedWidths { begin: pads[index], end: pads[index + axes.len()] };
    }
    Ok(widths)
}

/// The `Slice` that removes what negative widths remove; it lists no axes
/// when no width is negative.
fn crop(widths: &[SignedWidths], dims: &[Dim]) -> Result<SliceAttrs, ImportError> {
    let mut axes = Vec::new();
    for (index, (widths, dim)) in widths.iter().zip(dims).enumerate() {
        let (begin, end) = (removed(widths.begin), removed(widths.end));
        if begin == 0 && end == 0 {
            continue;
        }
        let length = dim
            .get()
            .checked_sub(begin)
            .and_then(|left| left.checked_sub(end))
            .ok_or_else(|| invalid("Pad removes more than the whole axis"))?;
        let axis = Axis::new(index);
        axes.push(SliceAxis {
            axis,
            start: begin,
            step: SliceStep::one(),
            length: Dim::new(length),
        });
    }
    Ok(SliceAttrs { axes })
}

fn removed(width: i64) -> u64 {
    width.min(0).unsigned_abs()
}

fn added(widths: SignedWidths) -> AxisPadding {
    AxisPadding { begin: grown(widths.begin), end: grown(widths.end) }
}

fn grown(width: i64) -> Padding {
    Padding::new(width.max(0).unsigned_abs())
}

/// The constant pad value: a scalar if the input is a known one-element
/// tensor (ONNX allows `[1]`), else the input, or a zero of `dtype` if
/// absent.
fn pad_value(context: &NodeContext<'_>, dtype: DType) -> Result<Operand, ImportError> {
    if !context.has_input(2) {
        return Ok(Operand::Constant(zero(dtype)?));
    }
    match context.entry(2)?.constant.as_ref() {
        Some(value) if value.shape().element_count().get() == 1 => {
            let (_, data) = value.clone().into_parts();
            Ok(Operand::Constant(
                HostTensor::new(Shape::scalar(), data)
                    .map_err(|error| invalid(error.to_string()))?,
            ))
        }
        _ => Ok(Operand::Input(2)),
    }
}

/// A scalar zero of `dtype`.
fn zero(dtype: DType) -> Result<HostTensor, ImportError> {
    let data: TensorData = match dtype {
        DType::F32 => vec![0.0_f32].into(),
        DType::F16 => vec![F16::from_f32(0.0)].into(),
        DType::Bf16 => vec![Bf16::from_f32(0.0)].into(),
        DType::F8E4M3Fn => vec![F8E4M3Fn::from_f32(0.0, Overflow::Saturate)].into(),
        DType::F8E5M2 => vec![F8E5M2::from_f32(0.0, Overflow::Saturate)].into(),
        DType::I8 => vec![0_i8].into(),
        DType::U8 => vec![0_u8].into(),
        DType::I32 => vec![0_i32].into(),
        DType::I64 => vec![0_i64].into(),
        DType::Bool => vec![false].into(),
    };
    HostTensor::new(Shape::scalar(), data).map_err(|error| invalid(error.to_string()))
}

/// `Trilu`, with `k` (default 0) known at import.
pub(super) fn trilu(context: &mut NodeContext<'_>) -> Result<Op, ImportError> {
    let part = if context.attrs.int_or("upper", 1)? == 0 {
        TriangularPart::Lower
    } else {
        TriangularPart::Upper
    };
    let diagonal = if context.has_input(1) {
        match context.ints(1)?.as_slice() {
            &[k] => k,
            _ => return Err(invalid("Trilu's k must be a scalar")),
        }
    } else {
        0
    };
    Ok(Op::Trilu(TriluAttrs { part, diagonal: DiagonalOffset::new(diagonal) }))
}
