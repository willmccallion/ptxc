//! Tolerances per element type and the mismatch report.

use ptxc_core::float::{F8E4M3Fn, Overflow};
use ptxc_core::{DType, Dim, ElementCount, HostTensor, Shape, TensorData};
use ptxc_testkit::compare::{ElementPosition, Mismatch, Tolerance, compare};

fn tensor(dims: &[u64], data: impl Into<TensorData>) -> HostTensor {
    HostTensor::new(Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap(), data.into())
        .unwrap()
}

fn f32s(values: &[f32]) -> HostTensor {
    tensor(&[u64::try_from(values.len()).unwrap()], values.to_vec())
}

fn passes(actual: &HostTensor, expected: &HostTensor, tolerance: Tolerance) -> bool {
    compare(actual, expected, tolerance).is_ok()
}

#[test]
fn defaults_follow_the_testing_table() {
    assert_eq!(Tolerance::for_dtype(DType::F32), Tolerance::Relative { atol: 1e-5, rtol: 1e-4 });
    assert_eq!(Tolerance::for_dtype(DType::F16), Tolerance::Relative { atol: 1e-3, rtol: 1e-2 });
    assert_eq!(Tolerance::for_dtype(DType::Bf16), Tolerance::Relative { atol: 1e-2, rtol: 2e-2 });
    assert_eq!(Tolerance::for_dtype(DType::F8E4M3Fn), Tolerance::Steps(1));
    assert_eq!(Tolerance::for_dtype(DType::F8E5M2), Tolerance::Steps(1));
    for dtype in [DType::I8, DType::U8, DType::I32, DType::I64, DType::Bool] {
        assert_eq!(Tolerance::for_dtype(dtype), Tolerance::Exact);
    }
    assert_eq!(Tolerance::ONNX_SUITE, Tolerance::Relative { atol: 1e-7, rtol: 1e-3 });
    assert_eq!(Tolerance::QUANTIZED, Tolerance::Steps(1));
}

#[test]
fn reductions_widen_relative_tolerances_by_the_root_of_their_length() {
    let length = ElementCount::new(16);

    assert_eq!(
        Tolerance::Relative { atol: 1e-5, rtol: 1e-4 }.for_reduction(length),
        Tolerance::Relative { atol: 4e-5, rtol: 4e-4 }
    );
    assert_eq!(Tolerance::Steps(1).for_reduction(length), Tolerance::Steps(1));
    assert_eq!(Tolerance::Exact.for_reduction(length), Tolerance::Exact);
}

#[test]
fn relative_tolerance_allows_atol_plus_rtol_times_the_reference() {
    let f32_default = Tolerance::for_dtype(DType::F32);
    let reference = f32s(&[1.0, 0.0]);

    assert!(passes(&f32s(&[1.0001, 0.000_009]), &reference, f32_default));
    assert!(!passes(&f32s(&[1.0002, 0.0]), &reference, f32_default));
    assert!(!passes(&f32s(&[1.0, 0.000_011]), &reference, f32_default));
}

#[test]
fn nans_match_each_other_and_infinities_match_themselves() {
    let tolerance = Tolerance::ONNX_SUITE;

    assert!(passes(
        &f32s(&[f32::NAN, f32::INFINITY]),
        &f32s(&[f32::NAN, f32::INFINITY]),
        tolerance
    ));
    assert!(!passes(&f32s(&[f32::NAN]), &f32s(&[1.0]), tolerance));
    assert!(!passes(&f32s(&[1.0]), &f32s(&[f32::NAN]), tolerance));
    assert!(!passes(&f32s(&[f32::INFINITY]), &f32s(&[f32::NEG_INFINITY]), tolerance));
    assert!(passes(&f32s(&[f32::NAN]), &f32s(&[f32::NAN]), Tolerance::Exact));
}

fn fp8(values: &[f32]) -> HostTensor {
    let encoded: Vec<F8E4M3Fn> =
        values.iter().map(|&v| F8E4M3Fn::from_f32(v, Overflow::Saturate)).collect();
    tensor(&[u64::try_from(values.len()).unwrap()], encoded)
}

#[test]
fn fp8_steps_count_representable_values_across_zero() {
    let one_step = Tolerance::Steps(1);
    let smallest = 2f32.powi(-9);

    assert!(passes(&fp8(&[1.125]), &fp8(&[1.0]), one_step));
    assert!(!passes(&fp8(&[1.25]), &fp8(&[1.0]), one_step));
    assert!(passes(&fp8(&[-0.0]), &fp8(&[0.0]), Tolerance::Exact));
    assert!(passes(&fp8(&[-smallest]), &fp8(&[0.0]), one_step));
    assert!(!passes(&fp8(&[-smallest]), &fp8(&[smallest]), one_step));
    assert!(passes(&fp8(&[-smallest]), &fp8(&[smallest]), Tolerance::Steps(2)));
}

#[test]
fn integers_are_exact_unless_quantized() {
    let five = tensor(&[1], vec![5_u8]);

    assert!(passes(&tensor(&[1], vec![6_u8]), &five, Tolerance::QUANTIZED));
    assert!(passes(&tensor(&[1], vec![4_u8]), &five, Tolerance::QUANTIZED));
    assert!(!passes(&tensor(&[1], vec![7_u8]), &five, Tolerance::QUANTIZED));
    assert!(!passes(&tensor(&[1], vec![6_u8]), &five, Tolerance::Exact));
    let huge = 1_i64 << 60;
    assert!(!passes(&tensor(&[1], vec![huge + 1]), &tensor(&[1], vec![huge]), Tolerance::Exact));
    assert!(!passes(&tensor(&[1], vec![true]), &tensor(&[1], vec![false]), Tolerance::Exact));
}

#[test]
fn element_types_and_shapes_must_agree() {
    let floats = f32s(&[1.0, 2.0]);

    let dtype = compare(&tensor(&[2], vec![1_i32, 2]), &floats, Tolerance::Exact).unwrap_err();
    let shape =
        compare(&tensor(&[1, 2], vec![1.0_f32, 2.0]), &floats, Tolerance::Exact).unwrap_err();

    assert_eq!(dtype, Mismatch::DType { actual: DType::I32, expected: DType::F32 });
    assert_eq!(shape.to_string(), "shape [Dim(1), Dim(2)], expected [Dim(2)]");
}

#[test]
fn the_report_counts_mismatches_and_finds_the_worst_by_each_measure() {
    let actual = tensor(&[2, 2], vec![1.0_f32, 12.0, 0.5, 4.0]);
    let expected = tensor(&[2, 2], vec![1.0_f32, 10.0, 0.25, 4.0]);

    let Mismatch::Values(report) = compare(&actual, &expected, Tolerance::ONNX_SUITE).unwrap_err()
    else {
        panic!()
    };

    assert_eq!((report.mismatched, report.total), (ElementCount::new(2), ElementCount::new(4)));
    assert_eq!(report.worst_absolute.position, ElementPosition(vec![0, 1]));
    assert_eq!(
        (report.worst_absolute.actual, report.worst_absolute.expected, report.worst_absolute.error),
        (12.0, 10.0, 2.0)
    );
    assert_eq!(report.worst_relative.position, ElementPosition(vec![1, 0]));
    assert_eq!(report.worst_relative.error, 1.0);
    assert_eq!(
        report.to_string(),
        "2 of 4 elements differ; worst absolute error 2 at [0, 1]: 12 vs expected 10; worst relative error 1 at [1, 0]: 0.5 vs expected 0.25"
    );
}

#[test]
fn errors_from_zero_or_nan_are_infinitely_relative() {
    let Mismatch::Values(report) =
        compare(&f32s(&[0.5, f32::NAN]), &f32s(&[0.0, 1.0]), Tolerance::ONNX_SUITE).unwrap_err()
    else {
        panic!()
    };

    assert_eq!(report.worst_absolute.position, ElementPosition(vec![1]));
    assert!(report.worst_absolute.error.is_infinite() && report.worst_relative.error.is_infinite());
}

#[test]
fn both_zeros_are_equal_in_every_float_type() {
    use ptxc_core::float::{Bf16, F8E5M2, F16};
    let zeros =
        |negative: HostTensor, positive: HostTensor| passes(&negative, &positive, Tolerance::Exact);

    assert!(zeros(f32s(&[-0.0]), f32s(&[0.0])));
    assert!(zeros(tensor(&[1], vec![F16::from_f32(-0.0)]), tensor(&[1], vec![F16::from_f32(0.0)])));
    assert!(zeros(
        tensor(&[1], vec![Bf16::from_f32(-0.0)]),
        tensor(&[1], vec![Bf16::from_f32(0.0)])
    ));
    let e5m2 = |v: f32| tensor(&[1], vec![F8E5M2::from_f32(v, Overflow::Saturate)]);
    assert!(zeros(e5m2(-0.0), e5m2(0.0)));
    assert!(!passes(&e5m2(-1.0), &e5m2(1.0), Tolerance::Steps(1)));
}

#[test]
fn nan_is_never_within_steps_of_a_number() {
    assert!(!passes(
        &tensor(&[1], vec![F8E4M3Fn::from_bits(0x7f)]),
        &fp8(&[448.0]),
        Tolerance::Steps(1)
    ));
}

#[test]
fn a_finite_result_never_matches_an_infinite_reference() {
    assert!(!passes(&f32s(&[1e30]), &f32s(&[f32::INFINITY]), Tolerance::ONNX_SUITE));
    assert!(!passes(&f32s(&[f32::INFINITY]), &f32s(&[1e30]), Tolerance::ONNX_SUITE));
}

#[test]
fn ties_report_the_first_worst_element_and_equal_infinities_have_no_error() {
    let actual = f32s(&[f32::INFINITY, 2.0, 3.0]);
    let expected = f32s(&[f32::INFINITY, 1.0, 2.0]);

    let Mismatch::Values(report) = compare(&actual, &expected, Tolerance::Exact).unwrap_err()
    else {
        panic!()
    };

    assert_eq!(report.worst_absolute.position, ElementPosition(vec![1]));
    assert_eq!(report.worst_absolute.error, 1.0);
}

#[test]
fn integer_reports_show_the_values() {
    let Mismatch::Values(report) =
        compare(&tensor(&[1], vec![7_i32]), &tensor(&[1], vec![5_i32]), Tolerance::Exact)
            .unwrap_err()
    else {
        panic!()
    };

    assert_eq!((report.worst_absolute.actual, report.worst_absolute.expected), (7.0, 5.0));
}
