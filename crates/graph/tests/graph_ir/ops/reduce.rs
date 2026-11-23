use crate::common::{f32_type, tensor_type};
use ptxc_core::{Axis, DType, Rank};
use ptxc_graph::{
    ArgKind, ArgReduceAttrs, CumSumAttrs, Inclusion, KeepDims, Op, ReduceAttrs, ReduceOp,
    ScanDirection, TensorType, TieBreak, TypeError, TypeRule,
};

const ALL_OPS: [ReduceOp; 10] = [
    ReduceOp::Sum,
    ReduceOp::Mean,
    ReduceOp::Prod,
    ReduceOp::Max,
    ReduceOp::Min,
    ReduceOp::SumSquare,
    ReduceOp::L1,
    ReduceOp::L2,
    ReduceOp::LogSum,
    ReduceOp::LogSumExp,
];

/// Axes resolved against a rank high enough for every test here.
fn axes(indices: &[i64]) -> Vec<Axis> {
    indices.iter().map(|&index| Axis::from_signed(index, Rank::new(8)).unwrap()).collect()
}

fn reduce(
    op: ReduceOp,
    indices: &[i64],
    keep_dims: KeepDims,
    input: &TensorType,
) -> Result<Vec<TensorType>, TypeError> {
    ReduceAttrs { op, axes: axes(indices), keep_dims }.infer(&[input])
}

fn argmax(index: i64, keep_dims: KeepDims) -> ArgReduceAttrs {
    ArgReduceAttrs { kind: ArgKind::Max, axis: axes(&[index])[0], keep_dims, tie: TieBreak::First }
}

#[test]
fn kept_axes_become_one_and_dropped_axes_disappear() {
    let input = f32_type(&[2, 3, 4]);

    for op in ALL_OPS {
        assert_eq!(
            reduce(op, &[1], KeepDims::Keep, &input),
            Ok(vec![f32_type(&[2, 1, 4])]),
            "{op:?}"
        );
        assert_eq!(reduce(op, &[1], KeepDims::Drop, &input), Ok(vec![f32_type(&[2, 4])]), "{op:?}");
    }
}

#[test]
fn several_axes_reduce_together_in_any_order() {
    let input = f32_type(&[2, 3, 4]);

    assert_eq!(reduce(ReduceOp::Sum, &[0, 2], KeepDims::Drop, &input), Ok(vec![f32_type(&[3])]));
    assert_eq!(
        reduce(ReduceOp::Sum, &[2, 0], KeepDims::Keep, &input),
        Ok(vec![f32_type(&[1, 3, 1])])
    );
}

#[test]
fn reducing_every_axis_without_keeping_gives_a_scalar() {
    let input = f32_type(&[2, 3]);

    assert_eq!(reduce(ReduceOp::Mean, &[0, 1], KeepDims::Drop, &input), Ok(vec![f32_type(&[])]));
}

#[test]
fn no_axes_reduces_nothing() {
    let input = f32_type(&[2, 3]);

    assert_eq!(reduce(ReduceOp::Sum, &[], KeepDims::Drop, &input), Ok(vec![input.clone()]));
}

#[test]
fn an_empty_axis_can_be_reduced() {
    assert_eq!(
        reduce(ReduceOp::Sum, &[0], KeepDims::Drop, &f32_type(&[0, 3])),
        Ok(vec![f32_type(&[3])])
    );
}

#[test]
fn axes_must_be_in_range_and_listed_once() {
    let input = f32_type(&[2, 3, 4]);
    let [third, fourth] = [axes(&[2])[0], axes(&[3])[0]];

    assert_eq!(
        reduce(ReduceOp::Sum, &[3], KeepDims::Keep, &input),
        Err(TypeError::AxisOutOfRange { axis: fourth, rank: Rank::new(3) })
    );
    assert_eq!(
        reduce(ReduceOp::Sum, &[2, 0, 2], KeepDims::Keep, &input),
        Err(TypeError::RepeatedAxis(third))
    );
}

#[test]
fn max_and_min_take_narrow_integers_and_bool() {
    for dtype in [DType::I8, DType::U8, DType::Bool] {
        let input = tensor_type(dtype, &[4]);
        for op in [ReduceOp::Max, ReduceOp::Min] {
            assert!(reduce(op, &[0], KeepDims::Drop, &input).is_ok(), "{op:?} on {dtype:?}");
        }
    }
}

#[test]
fn arithmetic_reductions_take_floats_and_wide_integers_only() {
    for op in ALL_OPS.into_iter().filter(|op| !matches!(op, ReduceOp::Max | ReduceOp::Min)) {
        assert!(reduce(op, &[0], KeepDims::Drop, &tensor_type(DType::I64, &[4])).is_ok(), "{op:?}");
        for dtype in [DType::I8, DType::Bool] {
            let error = reduce(op, &[0], KeepDims::Drop, &tensor_type(dtype, &[4]));
            assert_eq!(error, Err(TypeError::UnsupportedDType(dtype)), "{op:?}");
        }
    }
}

#[test]
fn arg_reductions_give_i64_indices() {
    let input = tensor_type(DType::U8, &[2, 3, 4]);

    assert_eq!(
        argmax(1, KeepDims::Keep).infer(&[&input]),
        Ok(vec![tensor_type(DType::I64, &[2, 1, 4])])
    );
    assert_eq!(
        argmax(2, KeepDims::Drop).infer(&[&input]),
        Ok(vec![tensor_type(DType::I64, &[2, 3])])
    );
}

#[test]
fn arg_reductions_reject_bool_and_out_of_range_axes() {
    let error = argmax(0, KeepDims::Keep).infer(&[&tensor_type(DType::Bool, &[4])]);

    assert_eq!(error, Err(TypeError::UnsupportedDType(DType::Bool)));
    assert!(matches!(
        argmax(1, KeepDims::Keep).infer(&[&f32_type(&[4])]),
        Err(TypeError::AxisOutOfRange { .. })
    ));
}

const fn cum_sum(axis: usize) -> Op {
    Op::CumSum(CumSumAttrs {
        axis: Axis::new(axis),
        inclusion: Inclusion::Inclusive,
        direction: ScanDirection::Forward,
    })
}

#[test]
fn cum_sum_keeps_its_operands_type() {
    for dtype in [DType::F32, DType::Bf16, DType::I32, DType::I64] {
        let input = tensor_type(dtype, &[2, 3]);

        assert_eq!(cum_sum(1).infer(&[&input]), Ok(vec![input.clone()]));
    }
}

#[test]
fn cum_sum_needs_a_summable_type_and_its_axis() {
    for dtype in [DType::I8, DType::U8, DType::Bool] {
        assert_eq!(
            cum_sum(0).infer(&[&tensor_type(dtype, &[2])]),
            Err(TypeError::UnsupportedDType(dtype))
        );
    }
    assert!(matches!(
        cum_sum(2).infer(&[&f32_type(&[2, 3])]),
        Err(TypeError::AxisOutOfRange { .. })
    ));
}
