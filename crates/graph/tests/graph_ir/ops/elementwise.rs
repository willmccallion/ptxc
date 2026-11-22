use crate::common::{f32_type, shape, tensor_type};
use ptxc_core::DType;
use ptxc_core::float::Overflow;
use ptxc_graph::{
    BinaryOp, CastAttrs, InfDetection, ModKind, Op, TensorType, TypeError, TypeRule, UnaryOp,
    VariadicOp,
};

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

#[test]
fn float_math_ops_reject_integers() {
    for op in [UnaryOp::Round, UnaryOp::Erf, UnaryOp::Sin, UnaryOp::Cos] {
        let operand = tensor_type(DType::I32, &[4]);
        assert_eq!(op.infer(&[&operand]), Err(TypeError::UnsupportedDType(DType::I32)), "{op:?}");
        assert!(op.infer(&[&f32_type(&[4])]).is_ok(), "{op:?}");
    }
}

#[test]
fn sign_accepts_unsigned_integers() {
    let operand = tensor_type(DType::U8, &[4]);

    assert_eq!(UnaryOp::Sign.infer(&[&operand]), Ok(vec![operand.clone()]));
}

#[test]
fn nan_and_inf_tests_give_bool_of_the_same_shape() {
    let operand = tensor_type(DType::F16, &[2, 3]);

    for op in [UnaryOp::IsNaN, UnaryOp::IsInf(InfDetection::Positive)] {
        assert_eq!(op.infer(&[&operand]), Ok(vec![tensor_type(DType::Bool, &[2, 3])]), "{op:?}");
    }
    let integer = tensor_type(DType::I32, &[2]);
    assert_eq!(UnaryOp::IsNaN.infer(&[&integer]), Err(TypeError::UnsupportedDType(DType::I32)));
}

#[test]
fn pow_keeps_the_base_type_and_takes_any_numeric_exponent() {
    let base = f32_type(&[2, 3]);
    let exponent = tensor_type(DType::I64, &[3]);

    assert_eq!(infer_binary(BinaryOp::Pow, &base, &exponent), Ok(f32_type(&[2, 3])));
}

#[test]
fn pow_rejects_narrow_integer_bases_and_bool_exponents() {
    let narrow = tensor_type(DType::U8, &[3]);
    let boolean = tensor_type(DType::Bool, &[3]);

    assert_eq!(
        infer_binary(BinaryOp::Pow, &narrow, &narrow),
        Err(TypeError::UnsupportedDType(DType::U8))
    );
    assert_eq!(
        infer_binary(BinaryOp::Pow, &f32_type(&[3]), &boolean),
        Err(TypeError::UnsupportedDType(DType::Bool))
    );
}

#[test]
fn integer_mod_takes_only_integers_and_fmod_takes_floats_too() {
    let float = f32_type(&[3]);
    let integer = tensor_type(DType::I32, &[3]);

    assert_eq!(
        infer_binary(BinaryOp::Mod(ModKind::Integer), &float, &float),
        Err(TypeError::UnsupportedDType(DType::F32))
    );
    assert_eq!(infer_binary(BinaryOp::Mod(ModKind::Integer), &integer, &integer), Ok(integer));
    assert_eq!(infer_binary(BinaryOp::Mod(ModKind::Fmod), &float, &float), Ok(float.clone()));
}

#[test]
fn ordered_comparisons_and_xor_give_bool() {
    let numbers = tensor_type(DType::I8, &[3]);
    let flags = tensor_type(DType::Bool, &[3]);
    let bools = tensor_type(DType::Bool, &[3]);

    for op in [BinaryOp::LessOrEqual, BinaryOp::GreaterOrEqual] {
        assert_eq!(infer_binary(op, &numbers, &numbers), Ok(bools.clone()), "{op:?}");
    }
    assert_eq!(infer_binary(BinaryOp::Xor, &flags, &flags), Ok(bools));
    assert!(infer_binary(BinaryOp::Xor, &numbers, &numbers).is_err());
}

#[test]
fn variadic_ops_broadcast_every_operand() {
    let operands = [f32_type(&[2, 1]), f32_type(&[3]), f32_type(&[1, 1])];
    let refs: Vec<&TensorType> = operands.iter().collect();

    for op in [VariadicOp::Max, VariadicOp::Min, VariadicOp::Sum, VariadicOp::Mean] {
        assert_eq!(op.infer(&refs), Ok(vec![f32_type(&[2, 3])]), "{op:?}");
    }
}

#[test]
fn variadic_ops_take_one_operand_but_not_none() {
    let operand = f32_type(&[4]);

    assert_eq!(VariadicOp::Mean.infer(&[&operand]), Ok(vec![operand.clone()]));
    assert_eq!(VariadicOp::Max.infer(&[]), Err(TypeError::NoOperands));
}

#[test]
fn variadic_operands_must_share_an_element_type() {
    let error =
        VariadicOp::Max.infer(&[&f32_type(&[2]), &f32_type(&[2]), &tensor_type(DType::F16, &[2])]);

    assert_eq!(error, Err(TypeError::MixedDTypes { lhs: DType::F32, rhs: DType::F16 }));
}

#[test]
fn sum_and_mean_take_only_floats_and_max_takes_integers() {
    let integer = tensor_type(DType::I32, &[2]);

    for op in [VariadicOp::Sum, VariadicOp::Mean] {
        assert_eq!(op.infer(&[&integer]), Err(TypeError::UnsupportedDType(DType::I32)), "{op:?}");
    }
    assert!(VariadicOp::Max.infer(&[&integer]).is_ok());
}

#[test]
fn where_broadcasts_condition_and_both_branches() {
    let condition = tensor_type(DType::Bool, &[2, 1]);
    let on_true = tensor_type(DType::I64, &[3]);
    let on_false = tensor_type(DType::I64, &[1]);

    let inferred = Op::Where.infer(&[&condition, &on_true, &on_false]);

    assert_eq!(inferred, Ok(vec![tensor_type(DType::I64, &[2, 3])]));
}

#[test]
fn where_needs_a_bool_condition_and_matching_branches() {
    let values = f32_type(&[3]);

    assert_eq!(
        Op::Where.infer(&[&values, &values, &values]),
        Err(TypeError::UnsupportedDType(DType::F32))
    );
    let condition = tensor_type(DType::Bool, &[3]);
    let other = tensor_type(DType::F16, &[3]);
    assert_eq!(
        Op::Where.infer(&[&condition, &values, &other]),
        Err(TypeError::MixedDTypes { lhs: DType::F32, rhs: DType::F16 })
    );
}

#[test]
fn cast_keeps_the_shape_and_takes_the_target_type() {
    let attrs = CastAttrs::new(DType::Bf16, Overflow::Saturate);

    assert_eq!(attrs.infer(&[&f32_type(&[2, 3])]), Ok(vec![tensor_type(DType::Bf16, &[2, 3])]));
}

#[test]
fn cast_keeps_overflow_only_for_fp8_targets() {
    let to_fp8 = CastAttrs::new(DType::F8E5M2, Overflow::NonFinite);
    let to_f16 = CastAttrs::new(DType::F16, Overflow::NonFinite);

    assert_eq!(to_fp8.overflow(), Overflow::NonFinite);
    assert_eq!(to_f16, CastAttrs::new(DType::F16, Overflow::Saturate));
    assert_eq!(to_f16.to(), DType::F16);
}
