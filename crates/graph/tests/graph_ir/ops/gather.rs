use crate::common::{f32_type, tensor_type};
use ptxc_core::{Axis, DType, Dim, Rank};
use ptxc_graph::{Op, TypeError, TypeRule};

fn axis(index: i64) -> Axis {
    Axis::from_signed(index, Rank::new(8)).unwrap()
}

#[test]
fn gather_replaces_its_axis_with_the_indices_shape() {
    let data = tensor_type(DType::Bool, &[4, 5, 6]);
    let indices = tensor_type(DType::I64, &[2, 3]);

    let gathered = Op::Gather(axis(1)).infer(&[&data, &indices]);

    assert_eq!(gathered, Ok(vec![tensor_type(DType::Bool, &[4, 2, 3, 6])]));
}

#[test]
fn gather_with_scalar_indices_removes_its_axis() {
    let gathered = Op::Gather(axis(0)).infer(&[&f32_type(&[4, 5]), &tensor_type(DType::I32, &[])]);

    assert_eq!(gathered, Ok(vec![f32_type(&[5])]));
}

#[test]
fn gather_needs_integer_indices_and_its_axis() {
    let data = f32_type(&[4]);

    assert_eq!(
        Op::Gather(axis(0)).infer(&[&data, &tensor_type(DType::I8, &[2])]),
        Err(TypeError::UnsupportedDType(DType::I8))
    );
    assert!(matches!(
        Op::Gather(axis(1)).infer(&[&data, &tensor_type(DType::I64, &[2])]),
        Err(TypeError::AxisOutOfRange { .. })
    ));
}

#[test]
fn gather_elements_takes_the_indices_shape() {
    let data = f32_type(&[3, 4]);
    let indices = tensor_type(DType::I64, &[3, 7]);

    assert_eq!(Op::GatherElements(axis(1)).infer(&[&data, &indices]), Ok(vec![f32_type(&[3, 7])]));
    let fewer = tensor_type(DType::I32, &[2, 4]);
    assert_eq!(Op::GatherElements(axis(1)).infer(&[&data, &fewer]), Ok(vec![f32_type(&[2, 4])]));
}

#[test]
fn gather_elements_indices_may_not_outgrow_data_off_the_axis() {
    let data = f32_type(&[3, 4]);
    let indices = tensor_type(DType::I64, &[5, 4]);

    let error = Op::GatherElements(axis(1)).infer(&[&data, &indices]);

    assert_eq!(
        error,
        Err(TypeError::DimMismatch { axis: axis(0), expected: Dim::new(3), actual: Dim::new(5) })
    );
}

#[test]
fn gather_elements_needs_equal_ranks_integer_indices_and_its_axis() {
    let data = f32_type(&[3, 4]);

    assert!(matches!(
        Op::GatherElements(axis(0)).infer(&[&data, &tensor_type(DType::I64, &[3])]),
        Err(TypeError::Rank { .. })
    ));
    assert!(matches!(
        Op::GatherElements(axis(0)).infer(&[&data, &f32_type(&[3, 4])]),
        Err(TypeError::UnsupportedDType(DType::F32))
    ));
    assert!(matches!(
        Op::GatherElements(axis(2)).infer(&[&data, &tensor_type(DType::I64, &[3, 4])]),
        Err(TypeError::AxisOutOfRange { .. })
    ));
}
