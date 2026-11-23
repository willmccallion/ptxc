use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16, Overflow};
use ptxc_core::{BytesError, DType, TensorData};

fn round_trip(data: &TensorData) -> TensorData {
    TensorData::from_le_bytes(data.dtype(), &data.to_le_bytes()).unwrap()
}

#[test]
fn every_type_round_trips_through_its_bytes() {
    let cases: Vec<TensorData> = vec![
        vec![1.5_f32, -0.0, f32::INFINITY].into(),
        vec![F16::from_f32(-2.5), F16::from_f32(65504.0)].into(),
        vec![Bf16::from_f32(3.0), Bf16::from_f32(-1e30)].into(),
        vec![F8E4M3Fn::from_f32(-448.0, Overflow::Saturate)].into(),
        vec![F8E5M2::from_f32(0.25, Overflow::Saturate)].into(),
        vec![-128_i8, 127].into(),
        vec![0_u8, 255].into(),
        vec![i32::MIN, 7].into(),
        vec![i64::MAX, -3].into(),
        vec![true, false].into(),
    ];
    for data in cases {
        assert_eq!(format!("{:?}", round_trip(&data)), format!("{data:?}"));
    }
}

#[test]
fn elements_are_little_endian_and_booleans_one_byte() {
    assert_eq!(TensorData::from(vec![0x0102_0304_i32]).to_le_bytes(), [4, 3, 2, 1]);
    assert_eq!(TensorData::from(vec![1.0_f32]).to_le_bytes(), [0, 0, 0x80, 0x3F]);
    assert_eq!(TensorData::from(vec![true, false]).to_le_bytes(), [1, 0]);
}

#[test]
fn bytes_that_are_not_whole_elements_are_rejected() {
    let error = TensorData::from_le_bytes(DType::I32, &[1, 2, 3, 4, 5]).unwrap_err();
    assert_eq!(error, BytesError::Length { dtype: DType::I32, length: 5 });
}

#[test]
fn a_boolean_byte_other_than_zero_or_one_is_rejected() {
    let error = TensorData::from_le_bytes(DType::Bool, &[1, 0, 2]).unwrap_err();
    assert_eq!(error, BytesError::Bool { index: 2, byte: 2 });
}
