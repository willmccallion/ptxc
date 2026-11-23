//! The element types each scalar operation takes and gives.

use crate::compute_build::shape;
use ptxc_core::{DType, Dim};
use ptxc_loop::{
    ArgKind, BinaryOp, ComputeBuilder, ComputeError, ComputeName, InfSign, ReduceOp, Scalar,
    TensorExpr, TensorName, TieBreak, UnaryOp, VarName,
};

/// The type an output of `expr(x)` has, where `x` is a `dtype` load inside
/// a reduction axis `k` that `expr` may reduce over.
fn type_of(
    dtype: DType,
    expr: impl Fn(TensorExpr, ptxc_loop::IterVarId) -> TensorExpr,
) -> Result<DType, ComputeError> {
    let mut b = ComputeBuilder::new(ComputeName::new("probe"));
    let x = b.input(TensorName::new("x"), dtype, shape(&[3]));
    let (k_axis, k) = b.reduce_axis(VarName::new("k"), Dim::new(3));
    let value = TensorExpr::Load { tensor: x, index: vec![k] };
    b.output(TensorName::new("out"), expr(value, k_axis));
    let def = b.finish()?;
    Ok(def.outputs()[0].dtype)
}

/// Wraps a per-element expression in a sum over `k`, so `k` is in scope.
fn summed(
    op: impl Fn(TensorExpr) -> TensorExpr,
) -> impl Fn(TensorExpr, ptxc_loop::IterVarId) -> TensorExpr {
    move |value, axis| TensorExpr::Reduce {
        op: ReduceOp::Max,
        axes: vec![axis],
        body: Box::new(op(value)),
    }
}

fn unary(op: UnaryOp, dtype: DType) -> Result<DType, ComputeError> {
    type_of(dtype, summed(|value| TensorExpr::Unary(op, Box::new(value))))
}

fn binary(op: BinaryOp, dtype: DType) -> Result<DType, ComputeError> {
    type_of(dtype, summed(|value| TensorExpr::Binary(op, Box::new(value.clone()), Box::new(value))))
}

const FLOATS: [DType; 3] = [DType::F32, DType::F16, DType::Bf16];
const INTEGERS: [DType; 4] = [DType::I8, DType::U8, DType::I32, DType::I64];
const STORAGE_ONLY: [DType; 2] = [DType::F8E4M3Fn, DType::F8E5M2];

#[test]
fn transcendental_ops_take_floats_only() {
    for op in
        [UnaryOp::Exp, UnaryOp::Log, UnaryOp::Sqrt, UnaryOp::Erf, UnaryOp::Tanh, UnaryOp::Round]
    {
        for dtype in FLOATS {
            assert_eq!(unary(op, dtype), Ok(dtype), "{op:?} {dtype:?}");
        }
        for dtype in INTEGERS.into_iter().chain(STORAGE_ONLY).chain([DType::Bool]) {
            assert_eq!(
                unary(op, dtype),
                Err(ComputeError::UnsupportedDType { op: op.name(), dtype })
            );
        }
    }
}

#[test]
fn sign_ops_take_any_number_and_tests_give_booleans() {
    for dtype in FLOATS.into_iter().chain(INTEGERS) {
        assert_eq!(unary(UnaryOp::Neg, dtype), Ok(dtype));
        assert_eq!(unary(UnaryOp::Abs, dtype), Ok(dtype));
    }
    assert_eq!(unary(UnaryOp::IsNaN, DType::F16).unwrap(), DType::Bool);
    assert_eq!(unary(UnaryOp::IsInf(InfSign::Positive), DType::F32).unwrap(), DType::Bool);
    assert!(unary(UnaryOp::IsNaN, DType::I32).is_err());
    assert_eq!(unary(UnaryOp::Not, DType::Bool), Ok(DType::Bool));
    assert!(unary(UnaryOp::Not, DType::I8).is_err());
    assert!(unary(UnaryOp::Neg, DType::Bool).is_err());
}

#[test]
fn arithmetic_keeps_its_type_and_comparisons_give_booleans() {
    for dtype in FLOATS.into_iter().chain(INTEGERS) {
        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Max,
            BinaryOp::TruncatedRem,
        ] {
            assert_eq!(binary(op, dtype), Ok(dtype), "{op:?} {dtype:?}");
        }
        for op in [BinaryOp::Equal, BinaryOp::Less, BinaryOp::GreaterOrEqual] {
            assert_eq!(binary(op, dtype), Ok(DType::Bool), "{op:?} {dtype:?}");
        }
    }
    for dtype in STORAGE_ONLY {
        assert!(binary(BinaryOp::Add, dtype).is_err());
    }
}

#[test]
fn some_binary_ops_take_one_kind_of_number() {
    assert_eq!(binary(BinaryOp::Pow, DType::F32), Ok(DType::F32));
    assert_eq!(binary(BinaryOp::Pow, DType::I32), Ok(DType::I32));
    assert!(binary(BinaryOp::Pow, DType::Bool).is_err());
    assert_eq!(binary(BinaryOp::FlooredRem, DType::I64), Ok(DType::I64));
    assert!(binary(BinaryOp::FlooredRem, DType::F32).is_err());
    assert_eq!(binary(BinaryOp::And, DType::Bool), Ok(DType::Bool));
    assert!(binary(BinaryOp::Xor, DType::I8).is_err());
    assert_eq!(binary(BinaryOp::Equal, DType::Bool), Ok(DType::Bool));
    assert!(binary(BinaryOp::Less, DType::Bool).is_err());
}

#[test]
fn reductions_keep_their_type_and_arg_reductions_give_i64() {
    let reduce = |op, dtype| {
        type_of(dtype, |value, axis| TensorExpr::Reduce {
            op,
            axes: vec![axis],
            body: Box::new(value),
        })
    };
    let arg = |dtype| {
        type_of(dtype, |value, axis| TensorExpr::ArgReduce {
            kind: ArgKind::Min,
            tie: TieBreak::Last,
            axis,
            body: Box::new(value),
        })
    };

    assert_eq!(reduce(ReduceOp::Sum, DType::I32), Ok(DType::I32));
    assert_eq!(reduce(ReduceOp::Min, DType::Bool), Ok(DType::Bool));
    assert!(reduce(ReduceOp::Prod, DType::Bool).is_err());
    assert_eq!(arg(DType::Bf16), Ok(DType::I64));
    assert_eq!(
        arg(DType::Bool),
        Err(ComputeError::UnsupportedDType { op: "argmin", dtype: DType::Bool })
    );
}

#[test]
fn a_cast_gives_its_target_from_any_type() {
    let cast = |from, to| {
        type_of(
            from,
            summed(move |value| TensorExpr::Cast {
                to,
                overflow: ptxc_core::float::Overflow::NonFinite,
                value: Box::new(value),
            }),
        )
    };

    assert_eq!(cast(DType::F8E4M3Fn, DType::F32), Ok(DType::F32));
    assert_eq!(
        cast(DType::Bool, DType::F8E5M2),
        Err(ComputeError::UnsupportedDType { op: "reduce_max", dtype: DType::F8E5M2 })
    );
    assert_eq!(cast(DType::I64, DType::Bool), Ok(DType::Bool));
}

#[test]
fn scalars_hold_values_their_type_can_represent() {
    assert!(Scalar::float(DType::I32, 1.0).is_none());
    assert_eq!(Scalar::float(DType::Bf16, 0.5).unwrap().as_f64(), Some(0.5));
    assert_eq!(Scalar::f32(0.1), Scalar::float(DType::F32, f64::from(0.1_f32)).unwrap());
    assert!(Scalar::int(DType::I8, 128).is_none());
    assert!(Scalar::int(DType::U8, -1).is_none());
    assert_eq!(Scalar::int(DType::U8, 255).unwrap().as_i64(), Some(255));
    assert!(Scalar::int(DType::F32, 1).is_none());
    assert_eq!(Scalar::int(DType::I8, -128).unwrap().as_i64(), Some(-128));
    assert_eq!(Scalar::bool(true).as_bool(), Some(true));
    assert_eq!(Scalar::bool(false).as_bool(), Some(false));
    assert_eq!(Scalar::int(DType::I32, 0).unwrap().as_bool(), None);
    assert_eq!(Scalar::bool(true).as_i64(), None);
    assert_eq!(Scalar::int(DType::I64, -3).unwrap().to_string(), "-3:i64");
    assert_eq!(Scalar::float(DType::F32, 1.5).unwrap().to_string(), "1.5:f32");
}

#[test]
fn every_type_has_its_identities_and_extremes() {
    let value = |scalar: Scalar| (scalar.as_f64(), scalar.as_i64(), scalar.as_bool());
    let int = |value: i64| (None, Some(value), None);
    let float = |value: f64| (Some(value), None, None);
    let cases = [
        (DType::I8, int(-128), int(127)),
        (DType::U8, int(0), int(255)),
        (DType::I32, int(i64::from(i32::MIN)), int(i64::from(i32::MAX))),
        (DType::I64, int(i64::MIN), int(i64::MAX)),
        (DType::F32, float(f64::NEG_INFINITY), float(f64::INFINITY)),
        (DType::F8E4M3Fn, float(-448.0), float(448.0)),
        (DType::F8E5M2, float(f64::NEG_INFINITY), float(f64::INFINITY)),
        (DType::Bool, (None, None, Some(false)), (None, None, Some(true))),
    ];
    for (dtype, lowest, highest) in cases {
        assert_eq!(value(Scalar::lowest(dtype)), lowest, "{dtype:?}");
        assert_eq!(value(Scalar::highest(dtype)), highest, "{dtype:?}");
        assert_eq!(Scalar::lowest(dtype).dtype(), dtype);
    }
    assert_eq!(value(Scalar::zero(DType::U8)), int(0));
    assert_eq!(value(Scalar::one(DType::Bf16)), float(1.0));
    assert_eq!(value(Scalar::one(DType::Bool)), (None, None, Some(true)));
}
