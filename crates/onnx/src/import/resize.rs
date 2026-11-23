//! Imports `Resize`, with its constant scales, sizes and region of interest.

use crate::context::NodeContext;
use crate::error::{ImportError, ImportErrorKind};
use crate::lowered::Lowered;
use ptxc_core::{Dim, TensorData};
use ptxc_graph::{
    Antialias, CoordinateTransform, F32Attr, NearestRounding, Op, OutsideWeights, ResizeAttrs,
    ResizeAxis, ResizeMode,
};

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    (op == "Resize").then(|| resize(context))
}

fn invalid(problem: impl Into<String>) -> ImportError {
    ImportErrorKind::Invalid { field: "Resize", problem: problem.into() }.into()
}

/// Every `Resize` attribute, read whether or not the mode uses it.
struct Attributes {
    mode: String,
    coordinates: String,
    nearest: String,
    cubic_a: f32,
    exclude_outside: i64,
    extrapolation: f32,
    antialias: i64,
    axes: Option<Vec<i64>>,
    policy: String,
}

fn read_attributes(context: &mut NodeContext<'_>) -> Result<Attributes, ImportError> {
    let attrs = &mut context.attrs;
    Ok(Attributes {
        mode: attrs.string_or("mode", "nearest")?,
        coordinates: attrs.string_or("coordinate_transformation_mode", "half_pixel")?,
        nearest: attrs.string_or("nearest_mode", "round_prefer_floor")?,
        cubic_a: attrs.float_or("cubic_coeff_a", -0.75)?,
        exclude_outside: attrs.int_or("exclude_outside", 0)?,
        extrapolation: attrs.float_or("extrapolation_value", 0.0)?,
        antialias: attrs.int_or("antialias", 0)?,
        axes: attrs.ints("axes")?.map(<[i64]>::to_vec),
        policy: attrs.string_or("keep_aspect_ratio_policy", "stretch")?,
    })
}

/// `Resize(X, roi?, scales?, sizes?)`. Each axis gets an output size and the
/// scale the coordinate transform uses, as ONNX derives them: from `scales`,
/// `floor(in * scale)`; from `sizes`, `size / in`, adjusted to one common
/// scale under `not_larger` and `not_smaller`.
fn resize(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let attributes = read_attributes(context)?;
    let dims = context.ty(0)?.shape().dims().to_vec();
    let rank = dims.len();
    let axes: Vec<usize> = match &attributes.axes {
        Some(axes) => axes
            .iter()
            .map(|&axis| context.axis(axis, ptxc_core::Rank::new(rank)).map(ptxc_core::Axis::index))
            .collect::<Result<_, _>>()?,
        None => (0..rank).collect(),
    };
    let mut resize_axes: Vec<ResizeAxis> =
        dims.iter().map(|&size| ResizeAxis { size, scale: F32Attr::new(1.0) }).collect();
    let scales = present_floats(context, 2)?;
    if let Some(scales) = scales.filter(|scales| !scales.is_empty()) {
        if attributes.policy != "stretch" {
            return Err(invalid("scales cannot be used with keep_aspect_ratio_policy"));
        }
        set_from_scales(&mut resize_axes, &axes, &scales)?;
    } else {
        let sizes = context.ints(3)?;
        set_from_sizes(&mut resize_axes, &axes, &sizes, &attributes.policy)?;
    }
    let attrs = ResizeAttrs {
        mode: mode(context, &attributes)?,
        coordinates: coordinates(context, &attributes, rank, &axes)?,
        antialias: if attributes.antialias == 0 { Antialias::Off } else { Antialias::On },
        axes: resize_axes,
    };
    Ok(Lowered::on_inputs(Op::Resize(attrs), 1))
}

/// A float input, if given.
fn present_floats(
    context: &NodeContext<'_>,
    index: usize,
) -> Result<Option<Vec<f32>>, ImportError> {
    if !context.has_input(index) {
        return Ok(None);
    }
    match context.constant(index)?.data() {
        TensorData::F32(values) => Ok(Some(values.clone())),
        _ => Err(invalid("scales and roi must be f32")),
    }
}

fn set_from_scales(
    axes: &mut [ResizeAxis],
    selected: &[usize],
    scales: &[f32],
) -> Result<(), ImportError> {
    if scales.len() != selected.len() {
        return Err(invalid("one scale per resized axis"));
    }
    for (&axis, &scale) in selected.iter().zip(scales) {
        let size = floor_to_size(to_f64(axes[axis].size)? * f64::from(scale))?;
        axes[axis] = ResizeAxis { size, scale: F32Attr::new(scale) };
    }
    Ok(())
}

fn set_from_sizes(
    axes: &mut [ResizeAxis],
    selected: &[usize],
    sizes: &[i64],
    policy: &str,
) -> Result<(), ImportError> {
    if sizes.len() != selected.len() {
        return Err(invalid("one size per resized axis"));
    }
    let ratios: Vec<f64> = selected
        .iter()
        .zip(sizes)
        .map(|(&axis, &size)| {
            Ok(to_f64(Dim::new(u64::try_from(size).map_err(|_| invalid("negative size"))?))?
                / to_f64(axes[axis].size)?)
        })
        .collect::<Result<_, ImportError>>()?;
    let common = match policy {
        "stretch" => None,
        "not_larger" => ratios.iter().copied().reduce(f64::min),
        "not_smaller" => ratios.iter().copied().reduce(f64::max),
        _ => return Err(invalid("unknown keep_aspect_ratio_policy")),
    };
    for ((&axis, &size), &ratio) in selected.iter().zip(sizes).zip(&ratios) {
        let (size, scale) = match common {
            None => (Dim::new(u64::try_from(size).unwrap_or(0)), ratio),
            Some(scale) => (round_to_size(scale * to_f64(axes[axis].size)?)?, scale),
        };
        axes[axis] = ResizeAxis { size, scale: F32Attr::new(narrow(scale)) };
    }
    Ok(())
}

fn to_f64(dim: Dim) -> Result<f64, ImportError> {
    u32::try_from(dim.get()).map(f64::from).map_err(|_| invalid("an axis is too long to resize"))
}

/// A ratio as the `f32` a resize attribute holds, rounded to nearest.
#[expect(
    clippy::cast_possible_truncation,
    reason = "narrowing a resize ratio to f32 rounds to nearest, as ONNX's f32 scales do"
)]
const fn narrow(ratio: f64) -> f32 {
    ratio as f32
}

/// `floor(value)` as a size, as ONNX computes sizes from scales.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is checked to be a finite non-negative whole number below 2^32"
)]
fn floor_to_size(value: f64) -> Result<Dim, ImportError> {
    let floor = value.floor();
    if !(0.0..4_294_967_296.0).contains(&floor) {
        return Err(invalid("a resized size is out of range"));
    }
    Ok(Dim::new(floor as u64))
}

/// `value` rounded half up, as ONNX's `round_int` for aspect-kept sizes.
fn round_to_size(value: f64) -> Result<Dim, ImportError> {
    floor_to_size(value + 0.5)
}

fn mode(context: &NodeContext<'_>, attributes: &Attributes) -> Result<ResizeMode, ImportError> {
    Ok(match attributes.mode.as_str() {
        "nearest" => ResizeMode::Nearest(match attributes.nearest.as_str() {
            "round_prefer_floor" => NearestRounding::RoundPreferFloor,
            "round_prefer_ceil" => NearestRounding::RoundPreferCeil,
            "floor" => NearestRounding::Floor,
            "ceil" => NearestRounding::Ceil,
            _ => return Err(context.attrs.unsupported("nearest_mode")),
        }),
        "linear" => ResizeMode::Linear,
        "cubic" => ResizeMode::Cubic {
            coefficient: F32Attr::new(attributes.cubic_a),
            outside: if attributes.exclude_outside == 0 {
                OutsideWeights::Include
            } else {
                OutsideWeights::Exclude
            },
        },
        _ => return Err(context.attrs.unsupported("mode")),
    })
}

fn coordinates(
    context: &NodeContext<'_>,
    attributes: &Attributes,
    rank: usize,
    selected: &[usize],
) -> Result<CoordinateTransform, ImportError> {
    Ok(match attributes.coordinates.as_str() {
        "half_pixel" => CoordinateTransform::HalfPixel,
        "half_pixel_symmetric" => CoordinateTransform::HalfPixelSymmetric,
        "pytorch_half_pixel" => CoordinateTransform::PytorchHalfPixel,
        "align_corners" => CoordinateTransform::AlignCorners,
        "asymmetric" => CoordinateTransform::Asymmetric,
        "tf_crop_and_resize" => {
            let roi = present_floats(context, 1)?
                .ok_or(ImportErrorKind::Missing("roi for tf_crop_and_resize"))?;
            if roi.len() != 2 * selected.len() {
                return Err(invalid("roi needs a start and an end for every resized axis"));
            }
            let mut regions = vec![[F32Attr::new(0.0), F32Attr::new(1.0)]; rank];
            for (index, &axis) in selected.iter().enumerate() {
                regions[axis] =
                    [F32Attr::new(roi[index]), F32Attr::new(roi[index + selected.len()])];
            }
            CoordinateTransform::TfCropAndResize {
                roi: regions,
                extrapolation: F32Attr::new(attributes.extrapolation),
            }
        }
        _ => return Err(context.attrs.unsupported("coordinate_transformation_mode")),
    })
}
