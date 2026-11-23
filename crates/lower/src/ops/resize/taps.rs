//! The input positions and weights each output position of a resized axis
//! reads, computed at lowering time from ONNX's definitions in f64.

use ptxc_core::convert::{int_to_f64, truncate};
use ptxc_graph::{
    Antialias, CoordinateTransform, NearestRounding, OutsideWeights, ResizeAttrs, ResizeMode,
};

/// The positions an output position reads and their weights, or `None`
/// where `tf_crop_and_resize` extrapolates.
pub(super) type Taps = Option<Vec<(i64, f64)>>;

/// One resized axis: its input and output lengths, its scale, and its
/// region of interest for `tf_crop_and_resize`.
pub(super) struct Axis {
    pub(super) input: i64,
    pub(super) output: i64,
    pub(super) scale: f64,
    pub(super) roi: Option<[f64; 2]>,
}

impl Axis {
    /// The taps of every output position.
    pub(super) fn taps(&self, attrs: &ResizeAttrs) -> Vec<Taps> {
        (0..self.output).map(|x| self.taps_at(attrs, x)).collect()
    }

    /// The input coordinate of output position `x`. `align_corners`,
    /// `pytorch_half_pixel` and `tf_crop_and_resize` use the unrounded
    /// output length `scale * input`, as ONNX's reference does.
    fn source(&self, transform: &CoordinateTransform, x: i64) -> f64 {
        let (x, input) = (int_to_f64(x), int_to_f64(self.input));
        let width = self.scale * input;
        let unit_width = width - 1.0 == 0.0;
        match transform {
            CoordinateTransform::HalfPixelSymmetric => {
                let adjustment = int_to_f64(self.output) / width;
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

    fn taps_at(&self, attrs: &ResizeAttrs, x: i64) -> Taps {
        let source = self.source(&attrs.coordinates, x);
        let outside = !(0.0..=int_to_f64(self.input) - 1.0).contains(&source);
        if matches!(attrs.coordinates, CoordinateTransform::TfCropAndResize { .. }) && outside {
            return None;
        }
        // An integral coordinate reads the position on its left, as ONNX's
        // reference prefers.
        let ratio = if source.fract() == 0.0 { 1.0 } else { source - source.floor() };
        let (start, mut weights) = coefficients(attrs.mode, attrs.antialias, ratio, self.scale);
        let base = truncate(source - ratio) + start;
        let positions: Vec<i64> = (base..).take(weights.len()).collect();
        if let ResizeMode::Cubic { outside: OutsideWeights::Exclude, .. } = attrs.mode {
            exclude_outside(&mut weights, &positions, self.input);
        }
        let last = self.input.checked_sub(1).filter(|&last| last >= 0)?;
        Some(positions.iter().map(|&p| p.clamp(0, last)).zip(weights).collect())
    }
}

/// The taps around a coordinate with fractional part `ratio`: the first
/// tap's offset from `floor(x - ratio)`, and each tap's weight.
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

/// Zeroes the weights of taps outside the input and renormalises the rest,
/// as `exclude_outside` asks.
fn exclude_outside(weights: &mut [f64], positions: &[i64], input: i64) {
    for (weight, &position) in weights.iter_mut().zip(positions) {
        if !(0..input).contains(&position) {
            *weight = 0.0;
        }
    }
    let total: f64 = weights.iter().sum();
    for weight in weights {
        *weight /= total;
    }
}
