//! `Resize` in every mode, coordinate transform and antialias setting, on
//! finite inputs.
//!
//! The reference multiplies zero-weight taps, so a NaN or an
//! infinity beside a selected element spreads there, which ptxc's direct
//! reads do not (D47).

use super::values::{Domain, tensor};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::select;
use ptxc_core::{DType, Dim, HostTensor};
use ptxc_graph::{
    Antialias, CoordinateTransform, F32Attr, NearestRounding, Op, OutsideWeights, ResizeAttrs,
    ResizeAxis, ResizeMode,
};

const TYPES: [DType; 7] =
    [DType::F32, DType::F16, DType::Bf16, DType::I8, DType::U8, DType::I32, DType::I64];

fn mode() -> impl Strategy<Value = ResizeMode> {
    let rounding = select(vec![
        NearestRounding::RoundPreferFloor,
        NearestRounding::RoundPreferCeil,
        NearestRounding::Floor,
        NearestRounding::Ceil,
    ]);
    let outside = prop_oneof![Just(OutsideWeights::Include), Just(OutsideWeights::Exclude)];
    let coefficient = select(vec![-0.75_f32, -0.5]).prop_map(F32Attr::new);
    prop_oneof![
        rounding.prop_map(ResizeMode::Nearest),
        Just(ResizeMode::Linear),
        (coefficient, outside)
            .prop_map(|(coefficient, outside)| ResizeMode::Cubic { coefficient, outside }),
    ]
}

/// Every transform; `tf_crop_and_resize` with regions that may reach past
/// the input, so some positions extrapolate.
fn coordinates(rank: usize) -> impl Strategy<Value = CoordinateTransform> {
    let bound = select(vec![-0.25_f32, 0.0, 0.3, 0.5, 1.0, 1.2]).prop_map(F32Attr::new);
    let crop = (vec([bound.clone(), bound], rank), select(vec![0.0_f32, 2.5])).prop_map(
        |(roi, extrapolation)| CoordinateTransform::TfCropAndResize {
            roi,
            extrapolation: F32Attr::new(extrapolation),
        },
    );
    prop_oneof![
        Just(CoordinateTransform::HalfPixel),
        Just(CoordinateTransform::HalfPixelSymmetric),
        Just(CoordinateTransform::PytorchHalfPixel),
        Just(CoordinateTransform::AlignCorners),
        Just(CoordinateTransform::Asymmetric),
        crop,
    ]
}

/// An axis resized from `input` elements: to a chosen size, with the
/// scale that implies, or by a chosen scale, with the size it floors to.
fn resized(input: u64) -> impl Strategy<Value = ResizeAxis> {
    let by_size = (0_u64..=6).prop_map(move |size| {
        let scale = if input == 0 { 1.0 } else { f64_ratio(size, input) };
        ResizeAxis { size: Dim::new(size), scale: F32Attr::new(scale) }
    });
    let by_scale = select(vec![0.5_f32, 0.75, 1.0, 1.5, 2.0, 3.0]).prop_map(move |scale| {
        let size = ptxc_core::convert::truncate(
            f64::from(scale) * ptxc_core::convert::int_to_f64(i64::try_from(input).unwrap()),
        );
        ResizeAxis { size: Dim::new(u64::try_from(size).unwrap()), scale: F32Attr::new(scale) }
    });
    prop_oneof![by_size, by_scale]
}

fn f64_ratio(size: u64, input: u64) -> f32 {
    let as_f64 = |n: u64| ptxc_core::convert::int_to_f64(i64::try_from(n).unwrap());
    ptxc_core::convert::f64_to_f32(as_f64(size) / as_f64(input))
}

/// `Resize` in every mode and coordinate transform, by scales or sizes.
pub fn resize_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let antialias = prop_oneof![Just(Antialias::Off), Just(Antialias::On)];
    (select(TYPES.to_vec()), vec(0_u64..=4, 1..=3), mode(), antialias)
        .prop_flat_map(|(dtype, dims, mode, antialias)| {
            let axes: Vec<_> = dims.iter().map(|&input| resized(input)).collect();
            (
                axes,
                coordinates(dims.len()),
                Just(mode),
                Just(antialias),
                tensor(dtype, dims, Domain::Finite),
            )
        })
        .prop_map(|(axes, coordinates, mode, antialias, input)| {
            (Op::Resize(ResizeAttrs { mode, coordinates, antialias, axes }), vec![input])
        })
}
