use ptxc_core::{
    Affine, Axis, Granularity, QuantDType, QuantError, QuantParams, Rank, Scale, ZeroPoint,
};

const fn affine(zero_point: i32) -> Affine {
    Affine { scale: Scale::new(0.5), zero_point: ZeroPoint::new(zero_point) }
}

fn per_tensor(dtype: QuantDType, zero_point: i32) -> Result<QuantParams, QuantError> {
    QuantParams::new(dtype, Granularity::PerTensor(affine(zero_point)))
}

const fn out_of_range(dtype: QuantDType, zero_point: i32) -> QuantError {
    QuantError::ZeroPointOutOfRange { zero_point: ZeroPoint::new(zero_point), dtype }
}

#[test]
fn zero_points_at_the_ends_of_each_integer_range_are_accepted() {
    let bounds = [
        (QuantDType::I8, -128, 127),
        (QuantDType::U8, 0, 255),
        (QuantDType::I32, i32::MIN, i32::MAX),
    ];

    for (dtype, low, high) in bounds {
        assert!(per_tensor(dtype, low).is_ok(), "{dtype:?} {low}");
        assert!(per_tensor(dtype, high).is_ok(), "{dtype:?} {high}");
    }
}

#[test]
fn zero_points_just_outside_each_integer_range_are_rejected() {
    let outside = [
        (QuantDType::I8, -129),
        (QuantDType::I8, 128),
        (QuantDType::U8, -1),
        (QuantDType::U8, 256),
    ];

    for (dtype, zero_point) in outside {
        assert_eq!(per_tensor(dtype, zero_point), Err(out_of_range(dtype, zero_point)));
    }
}

#[test]
fn fp8_accepts_only_a_zero_zero_point() {
    for dtype in [QuantDType::F8E4M3Fn, QuantDType::F8E5M2] {
        assert!(per_tensor(dtype, 0).is_ok());
        assert_eq!(per_tensor(dtype, 1), Err(out_of_range(dtype, 1)));
        assert_eq!(per_tensor(dtype, -1), Err(out_of_range(dtype, -1)));
    }
}

#[test]
fn per_axis_rejects_any_out_of_range_zero_point() {
    let axis = Axis::from_signed(0, Rank::new(2)).unwrap();
    let maps = vec![affine(0), affine(300), affine(5)];

    let error = QuantParams::new(QuantDType::U8, Granularity::PerAxis { axis, maps }).unwrap_err();

    assert_eq!(error, out_of_range(QuantDType::U8, 300));
}

#[test]
fn per_axis_keeps_its_maps_in_order() {
    let axis = Axis::from_signed(-1, Rank::new(2)).unwrap();
    let granularity = Granularity::PerAxis { axis, maps: vec![affine(1), affine(2)] };

    let params = QuantParams::new(QuantDType::I8, granularity.clone()).unwrap();

    assert_eq!((params.dtype(), params.granularity()), (QuantDType::I8, &granularity));
}

#[test]
fn scales_compare_by_bits() {
    assert_eq!(Scale::new(f32::NAN), Scale::new(f32::NAN));
    assert_ne!(Scale::new(0.0), Scale::new(-0.0));
    assert_eq!(Scale::new(0.25), Scale::new(0.25));
}
