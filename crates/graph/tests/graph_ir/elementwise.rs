use crate::common::{f32_type, shape, tensor_type};
use ptxc_core::DType;
use ptxc_graph::{BinaryOp, TensorType, TypeError, TypeRule, UnaryOp};

fn infer_binary(op: BinaryOp, lhs: &TensorType, rhs: &TensorType) -> Result<TensorType, TypeError> {
    op.infer(&[lhs, rhs]).map(|mut outputs| outputs.remove(0))
}

fn broadcast(lhs: &[u64], rhs: &[u64]) -> Result<TensorType, TypeError> {
    infer_binary(BinaryOp::Add, &f32_type(lhs), &f32_type(rhs))
}

#[test]
fn unary_ops_keep_their_operand_type() {
    let operand = f32_type(&[2, 3]);

    assert_eq!(UnaryOp::Neg.infer(&[&operand]), Ok(vec![operand.clone()]));
}

#[test]
fn unary_ops_reject_element_types_outside_their_class() {
    let rejected = [
        (UnaryOp::Exp, DType::I32),
        (UnaryOp::Sqrt, DType::Bool),
        (UnaryOp::Neg, DType::U8),
        (UnaryOp::Abs, DType::F8E4M3Fn),
        (UnaryOp::Not, DType::F32),
    ];

    for (op, dtype) in rejected {
        let operand = tensor_type(dtype, &[4]);
        assert_eq!(op.infer(&[&operand]), Err(TypeError::UnsupportedDType(dtype)), "{op:?}");
    }
}

#[test]
fn unary_ops_accept_element_types_in_their_class() {
    let accepted = [
        (UnaryOp::Exp, DType::Bf16),
        (UnaryOp::Neg, DType::I8),
        (UnaryOp::Abs, DType::U8),
        (UnaryOp::Not, DType::Bool),
    ];

    for (op, dtype) in accepted {
        let operand = tensor_type(dtype, &[4]);
        assert!(op.infer(&[&operand]).is_ok(), "{op:?} on {dtype:?}");
    }
}

#[test]
fn ops_reject_the_wrong_number_of_operands() {
    let operand = f32_type(&[4]);

    let error = BinaryOp::Add.infer(&[&operand]).unwrap_err();

    assert!(matches!(
        error,
        TypeError::Arity { expected, actual } if expected.get() == 2 && actual.get() == 1
    ));
    assert!(matches!(
        UnaryOp::Neg.infer(&[&operand, &operand]),
        Err(TypeError::Arity { expected, actual }) if expected.get() == 1 && actual.get() == 2
    ));
}

#[test]
fn binary_operands_must_share_an_element_type() {
    let error = infer_binary(BinaryOp::Mul, &f32_type(&[4]), &tensor_type(DType::F16, &[4]));

    assert_eq!(error, Err(TypeError::MixedDTypes { lhs: DType::F32, rhs: DType::F16 }));
}

#[test]
fn broadcasting_aligns_shapes_at_the_innermost_axis() {
    assert_eq!(broadcast(&[2, 3], &[3]), Ok(f32_type(&[2, 3])));
    assert_eq!(broadcast(&[2, 1, 4], &[3, 1]), Ok(f32_type(&[2, 3, 4])));
    assert_eq!(broadcast(&[], &[5]), Ok(f32_type(&[5])));
    assert_eq!(broadcast(&[5], &[]), Ok(f32_type(&[5])));
}

#[test]
fn broadcasting_a_zero_dim_against_one_gives_zero() {
    assert_eq!(broadcast(&[0, 3], &[1, 3]), Ok(f32_type(&[0, 3])));
    assert_eq!(broadcast(&[2, 1], &[1, 0]), Ok(f32_type(&[2, 0])));
}

#[test]
fn broadcasting_unequal_dims_without_a_one_is_rejected() {
    let error = broadcast(&[2, 3], &[4]);

    assert_eq!(error, Err(TypeError::NotBroadcastable { lhs: shape(&[2, 3]), rhs: shape(&[4]) }));
    assert!(broadcast(&[0], &[2]).is_err());
}

#[test]
fn comparisons_give_bool_with_the_broadcast_shape() {
    let lhs = tensor_type(DType::I64, &[2, 1]);
    let rhs = tensor_type(DType::I64, &[3]);

    for op in [BinaryOp::Less, BinaryOp::Greater, BinaryOp::Equal] {
        assert_eq!(infer_binary(op, &lhs, &rhs), Ok(tensor_type(DType::Bool, &[2, 3])), "{op:?}");
    }
}

#[test]
fn arithmetic_keeps_the_operand_element_type() {
    let operand = tensor_type(DType::U8, &[4]);

    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
        assert_eq!(infer_binary(op, &operand, &operand), Ok(operand.clone()), "{op:?}");
    }
}

#[test]
fn logical_ops_take_and_give_bool() {
    let bool_type = tensor_type(DType::Bool, &[4]);
    let int_type = tensor_type(DType::I32, &[4]);

    assert_eq!(infer_binary(BinaryOp::And, &bool_type, &bool_type), Ok(bool_type.clone()));
    assert_eq!(infer_binary(BinaryOp::Equal, &bool_type, &bool_type), Ok(bool_type.clone()));
    assert_eq!(
        infer_binary(BinaryOp::Or, &int_type, &int_type),
        Err(TypeError::UnsupportedDType(DType::I32))
    );
    assert_eq!(
        infer_binary(BinaryOp::Less, &bool_type, &bool_type),
        Err(TypeError::UnsupportedDType(DType::Bool))
    );
}

#[test]
fn arithmetic_rejects_fp8() {
    let operand = tensor_type(DType::F8E5M2, &[4]);

    assert_eq!(
        infer_binary(BinaryOp::Add, &operand, &operand),
        Err(TypeError::UnsupportedDType(DType::F8E5M2))
    );
}
