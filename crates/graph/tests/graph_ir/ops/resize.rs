use crate::common::{f32_type, tensor_type};
use ptxc_core::{DType, Dim, Rank};
use ptxc_graph::{
    Antialias, CoordinateTransform, F32Attr, NearestRounding, OutsideWeights, ResizeAttrs,
    ResizeAxis, ResizeMode, TypeError, TypeRule,
};

const fn axis(size: u64, scale: f32) -> ResizeAxis {
    ResizeAxis { size: Dim::new(size), scale: F32Attr::new(scale) }
}

const fn resize(
    mode: ResizeMode,
    coordinates: CoordinateTransform,
    axes: Vec<ResizeAxis>,
) -> ResizeAttrs {
    ResizeAttrs { mode, coordinates, antialias: Antialias::Off, axes }
}

const NEAREST: ResizeMode = ResizeMode::Nearest(NearestRounding::RoundPreferFloor);

#[test]
fn resize_gives_the_recorded_sizes() {
    let attrs = resize(
        NEAREST,
        CoordinateTransform::Asymmetric,
        vec![axis(1, 1.0), axis(3, 1.0), axis(8, 2.0), axis(2, 0.5)],
    );

    assert_eq!(attrs.infer(&[&f32_type(&[1, 3, 4, 4])]), Ok(vec![f32_type(&[1, 3, 8, 2])]));
}

#[test]
fn every_mode_keeps_the_input_type() {
    let cubic =
        ResizeMode::Cubic { coefficient: F32Attr::new(-0.75), outside: OutsideWeights::Exclude };
    let input = tensor_type(DType::U8, &[2, 4]);

    for mode in [NEAREST, ResizeMode::Linear, cubic] {
        let attrs = resize(mode, CoordinateTransform::HalfPixel, vec![axis(2, 1.0), axis(8, 2.0)]);
        assert_eq!(attrs.infer(&[&input]), Ok(vec![tensor_type(DType::U8, &[2, 8])]), "{mode:?}");
    }
}

#[test]
fn resize_needs_an_entry_per_axis_and_a_numeric_input() {
    let attrs = resize(NEAREST, CoordinateTransform::AlignCorners, vec![axis(4, 2.0)]);

    assert_eq!(
        attrs.infer(&[&f32_type(&[2, 2])]),
        Err(TypeError::Rank { expected: Rank::new(1), actual: Rank::new(2) })
    );
    assert_eq!(
        attrs.infer(&[&tensor_type(DType::Bool, &[2])]),
        Err(TypeError::UnsupportedDType(DType::Bool))
    );
}

#[test]
fn crop_and_resize_needs_a_region_per_axis() {
    let region = [F32Attr::new(0.0), F32Attr::new(1.0)];
    let transform = |roi: Vec<[F32Attr; 2]>| CoordinateTransform::TfCropAndResize {
        roi,
        extrapolation: F32Attr::new(0.0),
    };

    let fitting =
        resize(ResizeMode::Linear, transform(vec![region; 2]), vec![axis(2, 1.0), axis(6, 1.5)]);
    let short =
        resize(ResizeMode::Linear, transform(vec![region]), vec![axis(2, 1.0), axis(6, 1.5)]);

    assert_eq!(fitting.infer(&[&f32_type(&[2, 4])]), Ok(vec![f32_type(&[2, 6])]));
    assert_eq!(
        short.infer(&[&f32_type(&[2, 4])]),
        Err(TypeError::Rank { expected: Rank::new(2), actual: Rank::new(1) })
    );
}
