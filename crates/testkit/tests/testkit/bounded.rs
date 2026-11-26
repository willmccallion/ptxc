//! Comparison with the error bounds of f32 sums.

use ptxc_core::float::F16;
use ptxc_core::{Dim, ElementCount, HostTensor, Shape, TensorData};
use ptxc_testkit::compare::{ElementPosition, Mismatch, SumBound, compare_bounded};

fn tensor(dims: &[u64], data: impl Into<TensorData>) -> HostTensor {
    HostTensor::new(Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap(), data.into())
        .unwrap()
}

fn f32s(values: &[f32]) -> HostTensor {
    tensor(&[u64::try_from(values.len()).unwrap()], values.to_vec())
}

fn f16s(values: &[f32]) -> HostTensor {
    let encoded: Vec<F16> = values.iter().map(|&v| F16::from_f32(v)).collect();
    tensor(&[u64::try_from(values.len()).unwrap()], encoded)
}

fn bounded(
    actual: &HostTensor,
    estimate: &HostTensor,
    magnitude: &HostTensor,
    roundings: u32,
) -> Result<(), Mismatch> {
    compare_bounded(actual, SumBound { estimate, magnitude, roundings })
}

/// `1.0` moved by `steps` representable f32 values.
const fn one_plus_steps(steps: i32) -> f32 {
    f32::from_bits(1.0_f32.to_bits().checked_add_signed(steps).unwrap())
}

/// One rounding of a sum of magnitude 4 may move it by `γ₁ × 4`, just over
/// `2⁻²²`: two f32 steps above 1 and four below it, where steps are half as
/// wide.
#[test]
fn a_sum_lies_anywhere_its_interval_rounds_into() {
    let (estimate, magnitude) = (f32s(&[1.0, 1.0]), f32s(&[4.0, -4.0]));
    let check = |above, below| {
        let actual = f32s(&[one_plus_steps(above), one_plus_steps(below)]);
        bounded(&actual, &estimate, &magnitude, 1).is_ok()
    };

    assert!(check(2, -4));
    assert!(check(0, 0));
    assert!(!check(3, -4));
    assert!(!check(2, -5));
}

/// `5u` above 1 is halfway between two and three f32 steps, a tie rounding
/// to even at two; `γ₁ × 5` is a little more, so the bound reaches three.
#[test]
fn the_bound_is_gamma_n_a_little_more_than_n_u() {
    let (estimate, magnitude) = (f32s(&[1.0]), f32s(&[5.0]));

    assert!(bounded(&f32s(&[one_plus_steps(3)]), &estimate, &magnitude, 1).is_ok());
    assert!(bounded(&f32s(&[one_plus_steps(4)]), &estimate, &magnitude, 1).is_err());
}

#[test]
fn the_interval_grows_with_the_terms_not_with_the_sum() {
    let (estimate, magnitude) = (f32s(&[0.0]), f32s(&[1e6]));

    assert!(bounded(&f16s(&[0.125]), &estimate, &magnitude, 3).is_ok());
    assert!(bounded(&f16s(&[-0.125]), &estimate, &magnitude, 3).is_ok());
    assert!(bounded(&f16s(&[0.25]), &estimate, &magnitude, 3).is_err());
}

#[test]
fn an_exact_sum_must_round_to_the_result_s_type_exactly() {
    let (estimate, magnitude) = (f32s(&[1.0]), f32s(&[0.0]));

    assert!(bounded(&f16s(&[1.0]), &estimate, &magnitude, 4).is_ok());
    assert!(bounded(&f16s(&[1.0 + 2f32.powi(-10)]), &estimate, &magnitude, 4).is_err());
}

#[test]
fn a_nan_or_infinite_estimate_is_matched_exactly() {
    let (estimate, magnitude) = (f32s(&[f32::INFINITY, f32::NAN]), f32s(&[f32::INFINITY; 2]));

    assert!(bounded(&f32s(&[f32::INFINITY, f32::NAN]), &estimate, &magnitude, 1).is_ok());
    assert!(bounded(&f32s(&[f32::MAX, f32::NAN]), &estimate, &magnitude, 1).is_err());
    assert!(bounded(&f32s(&[f32::INFINITY, 0.0]), &estimate, &magnitude, 1).is_err());
}

#[test]
fn the_report_names_the_first_element_outside_and_its_interval() {
    let (estimate, magnitude) = (f32s(&[1.0; 3]), f32s(&[0.0; 3]));

    let Mismatch::Bounds(report) =
        bounded(&f32s(&[1.0, 2.0, 3.0]), &estimate, &magnitude, 1).unwrap_err()
    else {
        panic!()
    };

    assert_eq!((report.outside, report.total), (ElementCount::new(2), ElementCount::new(3)));
    assert_eq!(report.position, ElementPosition(vec![1]));
    assert_eq!((report.actual, report.low, report.high), (2.0, 1.0, 1.0));
    assert_eq!(
        report.to_string(),
        "2 of 3 elements lie outside their sums' error bounds; the first, at [1], is 2, not in [1, 1]"
    );
}

#[test]
fn estimates_and_magnitudes_must_have_the_result_s_shape() {
    let actual = tensor(&[1, 2], vec![1.0_f32, 2.0]);
    let wide = f32s(&[1.0, 2.0]);

    let estimate = bounded(&actual, &wide, &actual, 1).unwrap_err();
    let magnitude = bounded(&actual, &actual, &wide, 1).unwrap_err();

    for mismatch in [estimate, magnitude] {
        assert_eq!(mismatch.to_string(), "shape [Dim(1), Dim(2)], expected [Dim(2)]");
    }
}
