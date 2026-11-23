use crate::elementwise::lowered;
use ptxc_core::{DType, Dim};
use ptxc_graph::{
    Antialias, CoordinateTransform, F32Attr, NearestRounding, Op, OutsideWeights, ResizeAttrs,
    ResizeAxis, ResizeMode,
};
use ptxc_testkit::golden;

const fn axis(size: u64, scale: f32) -> ResizeAxis {
    ResizeAxis { size: Dim::new(size), scale: F32Attr::new(scale) }
}

const fn resize(mode: ResizeMode, coordinates: CoordinateTransform, axes: Vec<ResizeAxis>) -> Op {
    Op::Resize(ResizeAttrs { mode, coordinates, antialias: Antialias::Off, axes })
}

#[test]
fn unchanged_axes_read_directly_and_nearest_axes_read_a_position_table() {
    let nearest = ResizeMode::Nearest(NearestRounding::RoundPreferFloor);
    let definitions = [
        lowered(
            resize(nearest, CoordinateTransform::Asymmetric, vec![axis(2, 1.0), axis(6, 2.0)]),
            &[(DType::U8, &[2, 3])],
        ),
        lowered(
            resize(
                ResizeMode::Linear,
                CoordinateTransform::HalfPixel,
                vec![axis(2, 1.0), axis(5, 2.5)],
            ),
            &[(DType::F16, &[2, 2])],
        ),
    ];

    golden::check("resize/reads.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn integers_truncate_and_crops_extrapolate() {
    let cubic =
        ResizeMode::Cubic { coefficient: F32Attr::new(-0.75), outside: OutsideWeights::Exclude };
    let crop = CoordinateTransform::TfCropAndResize {
        roi: vec![[F32Attr::new(-0.5), F32Attr::new(1.0)]],
        extrapolation: F32Attr::new(7.0),
    };
    let definitions = [
        lowered(
            resize(cubic, CoordinateTransform::AlignCorners, vec![axis(3, 0.75)]),
            &[(DType::I32, &[4])],
        ),
        lowered(
            resize(ResizeMode::Linear, crop.clone(), vec![axis(3, 1.0)]),
            &[(DType::F32, &[3])],
        ),
        lowered(resize(ResizeMode::Linear, crop, vec![axis(2, 1.0)]), &[(DType::I8, &[0])]),
    ];

    golden::check("resize/arithmetic.txt", &definitions.join("\n")).unwrap();
}
