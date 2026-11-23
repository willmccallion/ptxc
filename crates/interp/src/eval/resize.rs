//! `Resize`, as ONNX's reference computes it: per axis, the source
//! coordinate's nearest input positions (edge-padded) weighted by the mode's
//! coefficients, combined over axes as a tensor product.

use crate::error::{InterpError, Malformed};
use crate::eval::{first, round_to_output};
use crate::extents::Extents;
use crate::numeric::{Elements, count_to_f64, int_to_f64, saturate_ints, signed, truncate, widen};
use ptxc_core::{HostTensor, TensorData};
use ptxc_graph::{
    Antialias, CoordinateTransform, NearestRounding, OutsideWeights, ResizeAttrs, ResizeMode,
    TensorType,
};

/// The input positions and weights one output position reads along an axis,
/// or `None` where `tf_crop_and_resize` extrapolates.
type Taps = Option<Vec<(usize, f64)>>;

/// One axis's lengths, scale and region of interest.
struct Axis {
    input: usize,
    output: usize,
    scale: f64,
    roi: Option<[f64; 2]>,
}

impl Axis {
    /// The input coordinate of output position `x`, by the transform, as
    /// ONNX's reference computes it: `align_corners`, `pytorch_half_pixel`
    /// and `tf_crop_and_resize` use the unrounded output length
    /// `scale * input`.
    fn source(&self, transform: &CoordinateTransform, x: usize) -> f64 {
        let (x, input) = (count_to_f64(x), count_to_f64(self.input));
        let width = self.scale * input;
        let unit_width = width - 1.0 == 0.0;
        match transform {
            CoordinateTransform::HalfPixelSymmetric => {
                let adjustment = count_to_f64(self.output) / width;
                (input / 2.0).mul_add(1.0 - adjustment, (x + 0.5) / self.scale - 0.5)
            }
            // NOTE: the spec gives 0 for a one-element output; ONNX's
            // reference, which produces the test data, gives -0.5.
            CoordinateTransform::PytorchHalfPixel if unit_width => -0.5,
            CoordinateTransform::HalfPixel | CoordinateTransform::PytorchHalfPixel => {
                (x + 0.5) / self.scale - 0.5
            }
            CoordinateTransform::AlignCorners if unit_width => 0.0,
            CoordinateTransform::AlignCorners => x * (input - 1.0) / (width - 1.0),
            CoordinateTransform::Asymmetric => x / self.scale,
            CoordinateTransform::TfCropAndResize { .. } => {
                let [start, end] = self.roi.unwrap_or([0.0, 1.0]);
                let offset = if unit_width {
                    (end - start) * (input - 1.0) / 2.0
                } else {
                    x * (end - start) * (input - 1.0) / (width - 1.0)
                };
                start.mul_add(input - 1.0, offset)
            }
        }
    }
}

/// The taps around a source coordinate with fractional part `ratio` (1 for
/// an integral coordinate, as ONNX's reference prefers the position on the
/// left): the offset of the first tap from `floor(x - ratio)`, and each
/// tap's weight.
fn coefficients(mode: ResizeMode, antialias: Antialias, ratio: f64, scale: f64) -> (i64, Vec<f64>) {
    match (mode, antialias) {
        (ResizeMode::Nearest(rounding), _) => (0, nearest(rounding, ratio)),
        (ResizeMode::Linear, Antialias::Off) => (0, vec![1.0 - ratio, ratio]),
        (ResizeMode::Linear, Antialias::On) => {
            stretched(ratio, scale, 1.0, |x| (1.0 - x.abs()).max(0.0))
        }
        (ResizeMode::Cubic { coefficient, .. }, antialias) => {
            let a = f64::from(coefficient.get());
            let scale = if antialias == Antialias::On { scale } else { 1.0 };
            stretched(ratio, scale, 2.0, move |x| cubic(a, x))
        }
    }
}

/// Two taps, `[left, right]`, one of them chosen by the rounding.
fn nearest(rounding: NearestRounding, ratio: f64) -> Vec<f64> {
    let right = ratio >= 1.0
        || match rounding {
            NearestRounding::RoundPreferFloor => ratio > 0.5,
            NearestRounding::RoundPreferCeil => ratio >= 0.5,
            NearestRounding::Floor => false,
            NearestRounding::Ceil => true,
        };
    if right { vec![0.0, 1.0] } else { vec![1.0, 0.0] }
}

/// Keys' cubic kernel with coefficient `a`.
fn cubic(a: f64, x: f64) -> f64 {
    let x = x.abs();
    if x <= 1.0 {
        ((a + 2.0).mul_add(x, -(a + 3.0)) * x).mul_add(x, 1.0)
    } else if x < 2.0 {
        a.mul_add(x, -5.0 * a).mul_add(x, 8.0 * a).mul_add(x, -4.0 * a)
    } else {
        0.0
    }
}

/// A kernel of half-width `support`, stretched by `1 / scale` when
/// downsampling, sampled at taps `start..2 - start` and normalised.
fn stretched(ratio: f64, scale: f64, support: f64, kernel: impl Fn(f64) -> f64) -> (i64, Vec<f64>) {
    let scale = scale.min(1.0);
    let start = truncate((-support / scale).floor() + 1.0);
    let weights: Vec<f64> =
        (start..2 - start).map(|i| kernel(scale * (int_to_f64(i) - ratio))).collect();
    let total: f64 = weights.iter().sum();
    (start, weights.into_iter().map(|w| w / total).collect())
}

impl Axis {
    /// The taps of every output position along this axis.
    fn taps(&self, attrs: &ResizeAttrs) -> Vec<Taps> {
        (0..self.output).map(|x| self.taps_at(attrs, x)).collect()
    }

    fn taps_at(&self, attrs: &ResizeAttrs, x: usize) -> Taps {
        let source = self.source(&attrs.coordinates, x);
        let last_coordinate = count_to_f64(self.input) - 1.0;
        let outside = !(0.0..=last_coordinate).contains(&source);
        if matches!(attrs.coordinates, CoordinateTransform::TfCropAndResize { .. }) && outside {
            return None;
        }
        let fraction = source.fract();
        let ratio = if fraction == 0.0 { 1.0 } else { source - source.floor() };
        let (start, mut weights) = coefficients(attrs.mode, attrs.antialias, ratio, self.scale);
        let base = truncate(source - ratio);
        let positions: Vec<i128> =
            (0..weights.len()).map(|i| i128::from(base + start) + signed(i)).collect();
        if let ResizeMode::Cubic { outside: OutsideWeights::Exclude, .. } = attrs.mode {
            exclude_outside(&mut weights, &positions, self.input);
        }
        let last = self.input.checked_sub(1)?;
        let clamp = |p: i128| usize::try_from(p.max(0)).map_or(last, |p| p.min(last));
        Some(positions.iter().map(|&p| clamp(p)).zip(weights).collect())
    }
}

/// Zero the weights of taps outside the input and renormalise the rest, as
/// `exclude_outside` asks.
fn exclude_outside(weights: &mut [f64], positions: &[i128], input: usize) {
    for (weight, &position) in weights.iter_mut().zip(positions) {
        if !(0..signed(input)).contains(&position) {
            *weight = 0.0;
        }
    }
    let total: f64 = weights.iter().sum();
    for weight in weights {
        *weight /= total;
    }
}

/// `Resize`: every output element is the tensor product of its axes' taps,
/// or the extrapolation value where any axis extrapolates. Integer inputs
/// truncate and saturate the interpolated value.
pub(super) fn resize(
    attrs: &ResizeAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let (extents, out) = (Extents::of(input.shape())?, Extents::of(output.shape())?);
    let axes = axes(attrs, &extents, &out);
    let taps: Vec<Vec<Taps>> = axes.iter().map(|axis| axis.taps(attrs)).collect();
    let extrapolation = match &attrs.coordinates {
        CoordinateTransform::TfCropAndResize { extrapolation, .. } => {
            f64::from(extrapolation.get())
        }
        _ => 0.0,
    };
    let values: Vec<f64> = match widen(input.data()) {
        Elements::Float(v) => v.into_iter().map(f64::from).collect(),
        Elements::Int(v) => v.into_iter().map(int_to_f64).collect(),
        Elements::Bool(_) => return Err(Malformed::DType(input.dtype()).into()),
    };
    let resized: Vec<f64> = out
        .indices()
        .map(|index| {
            let chosen: Option<Vec<&Vec<(usize, f64)>>> =
                index.iter().zip(&taps).map(|(&i, axis)| axis[i].as_ref()).collect();
            chosen.map_or(extrapolation, |chosen| weighted_sum(&values, &extents, &chosen))
        })
        .collect();
    match widen(input.data()) {
        Elements::Int(_) => {
            saturate_ints(&resized.into_iter().map(truncate).collect::<Vec<_>>(), output.dtype())
        }
        _ => round_to_output(&resized, output),
    }
}

fn axes(attrs: &ResizeAttrs, input: &Extents, output: &Extents) -> Vec<Axis> {
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
        .enumerate()
        .map(|(a, (axis, roi))| Axis {
            input: input.dims()[a],
            output: output.dims()[a],
            scale: f64::from(axis.scale.get()),
            roi,
        })
        .collect()
}

/// The sum over every combination of one tap per axis of the product of
/// their weights times the input element they select.
fn weighted_sum(values: &[f64], extents: &Extents, chosen: &[&Vec<(usize, f64)>]) -> f64 {
    let combinations = Extents::from_dims(chosen.iter().map(|taps| taps.len()).collect());
    combinations
        .indices()
        .map(|pick| {
            let (offset, weight) = pick.iter().zip(chosen).zip(extents.strides()).fold(
                (0, 1.0),
                |(offset, weight), ((&k, taps), stride)| {
                    let (position, w) = taps[k];
                    (offset + position * stride, weight * w)
                },
            );
            weight * values[offset]
        })
        .sum()
}
