use ptxc_core::float::F16;
use ptxc_core::{DType, Dim, ElementCount, HostTensor, Shape, TensorData, TensorError};

fn shape(extents: &[u64]) -> Shape {
    Shape::new(extents.iter().copied().map(Dim::new).collect()).unwrap()
}

#[test]
fn data_dtype_follows_the_element_type() {
    let cases = [
        (TensorData::from(vec![0.0_f32]), DType::F32),
        (TensorData::from(vec![F16::from_f32(1.0)]), DType::F16),
        (TensorData::from(vec![0_u8]), DType::U8),
        (TensorData::from(vec![0_i64]), DType::I64),
        (TensorData::from(vec![true]), DType::Bool),
    ];

    for (data, dtype) in cases {
        assert_eq!(data.dtype(), dtype);
    }
}

#[test]
fn tensor_keeps_its_shape_and_elements() {
    let tensor = HostTensor::new(shape(&[2, 3]), vec![1_i32, 2, 3, 4, 5, 6].into()).unwrap();

    assert_eq!((tensor.shape(), tensor.dtype()), (&shape(&[2, 3]), DType::I32));
    let TensorData::I32(elements) = tensor.data() else { panic!("expected i32 data") };
    assert_eq!(elements, &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn scalar_tensor_holds_one_element() {
    assert!(HostTensor::new(Shape::scalar(), vec![7.0_f32].into()).is_ok());
}

#[test]
fn empty_tensor_holds_no_elements() {
    assert!(HostTensor::new(shape(&[3, 0]), Vec::<f32>::new().into()).is_ok());
}

#[test]
fn data_shorter_or_longer_than_the_shape_is_rejected() {
    for data_len in [5, 7] {
        let data = TensorData::from(vec![0_u8; data_len]);

        let error = HostTensor::new(shape(&[2, 3]), data).unwrap_err();

        let expected =
            TensorError::LengthMismatch { shape_elements: ElementCount::new(6), data_len };
        assert_eq!(error, expected);
    }
}

#[test]
fn into_parts_returns_the_shape_and_data() {
    let tensor = HostTensor::new(shape(&[2]), vec![false, true].into()).unwrap();

    let (parts_shape, data) = tensor.into_parts();

    assert_eq!(parts_shape, shape(&[2]));
    assert!(matches!(data, TensorData::Bool(elements) if elements == [false, true]));
}
