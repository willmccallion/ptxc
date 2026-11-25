//! Loads that read padding outside their tensor: what a definition
//! accepts, how an unstaged one lowers, and that a staged tile holds the
//! padding, zero or not.

use crate::compute_build::{load, shape};
use crate::interp::{ints, run_default, tensor};
use crate::lower::lower;
use crate::schedule_support::open;
use ptxc_core::{DType, Dim, Idx};
use ptxc_loop::{
    ComputeBuilder, ComputeDef, ComputeError, ComputeName, ExprId, Primitive, ReduceOp, Scalar,
    ScheduleError, SplitFactor, TensorExpr, TensorName, TensorParamId, ThreadAxis, VarName,
    print_kernel, run,
};
use ptxc_testkit::golden;
use std::sync::Arc;

/// `x[i - 1]`, padded with `padding` where `i - 1` leaves `x`.
fn shifted(
    compute: &mut ComputeBuilder,
    x: TensorParamId,
    i: ExprId,
    padding: Scalar,
) -> TensorExpr {
    let minus_one = compute.exprs().constant(-1);
    let index = compute.exprs().add(i, minus_one);
    TensorExpr::Load { tensor: x, index: vec![index], padding: Some(padding) }
}

/// `out[i] = x[i - 1]` over `i` in `[0, 5)`, `x` an `i32[3]` padded with
/// `padding`.
fn shift(padding: i64) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("shift"));
    let x = compute.input(TensorName::new("x"), DType::I32, shape(&[3]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(5));
    let read = shifted(&mut compute, x, i, Scalar::int(DType::I32, padding).unwrap());
    compute.output(TensorName::new("out"), read);
    Arc::new(compute.finish().unwrap())
}

#[test]
fn a_padded_load_may_leave_its_tensor_but_an_unpadded_one_may_not() {
    let mut unpadded = ComputeBuilder::new(ComputeName::new("shift"));
    let x = unpadded.input(TensorName::new("x"), DType::I32, shape(&[3]));
    let (_, i) = unpadded.spatial(VarName::new("i"), Dim::new(5));
    unpadded.output(TensorName::new("out"), load(x, &[i]));

    let rejected = unpadded.finish().unwrap_err();

    assert!(matches!(rejected, ComputeError::IndexOutOfBounds { .. }), "{rejected:?}");
    assert_eq!(shift(7).inputs().len(), 1);
}

#[test]
fn padding_has_its_tensors_element_type() {
    let mut compute = ComputeBuilder::new(ComputeName::new("shift"));
    let x = compute.input(TensorName::new("x"), DType::I32, shape(&[3]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(5));
    let read = shifted(&mut compute, x, i, Scalar::f32(0.0));
    compute.output(TensorName::new("out"), read);

    let rejected = compute.finish().unwrap_err();

    let expected = ComputeError::PaddingType {
        tensor: TensorName::new("x"),
        expected: DType::I32,
        actual: DType::F32,
    };
    assert_eq!(rejected, expected);
}

#[test]
fn an_unstaged_padded_load_reads_the_clamped_element_or_its_padding() {
    let compute = shift(7);
    let (schedule, _) = open(&compute);

    let kernel = lower(schedule);
    let out = run(&kernel, &[tensor(&[3], vec![1_i32, 2, 3])]).unwrap();

    golden::check("lower/padded_load.txt", &print_kernel(&kernel)).unwrap();
    assert_eq!(ints(&out[0]), [7, 1, 2, 3, 7]);
}

#[test]
fn a_padded_load_of_an_empty_tensor_reads_only_its_padding() {
    let mut compute = ComputeBuilder::new(ComputeName::new("empty"));
    let x = compute.input(TensorName::new("x"), DType::I32, shape(&[0]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(2));
    let read = shifted(&mut compute, x, i, Scalar::int(DType::I32, 5).unwrap());
    compute.output(TensorName::new("out"), read);
    let compute = Arc::new(compute.finish().unwrap());
    let (schedule, _) = open(&compute);

    let out = run(&lower(schedule), &[tensor(&[0], Vec::<i32>::new())]).unwrap();

    assert_eq!(ints(&out[0]), [5, 5]);
}

/// `out[i] = sum over k of x[i + k - 1]`, `x` an `i32[12]` padded with
/// `padding`: a 3-tap window sliding past both ends.
fn window(padding: i64) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("window"));
    let x = compute.input(TensorName::new("x"), DType::I32, shape(&[12]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(12));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(3));
    let tap = compute.exprs().add(i, k);
    let read = shifted(&mut compute, x, tap, Scalar::int(DType::I32, padding).unwrap());
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body: Box::new(read) };
    compute.output(TensorName::new("out"), sum);
    Arc::new(compute.finish().unwrap())
}

/// `window` in blocks of 4 threads, its input staged per block: a tile of
/// 6 elements overhanging `x` at both ends.
fn staged_window(padding: i64) -> (Arc<ComputeDef>, ptxc_loop::Schedule<ptxc_loop::Open>) {
    let compute = window(padding);
    let (mut schedule, nest) = open(&compute);
    let parts = schedule.split(nest[0], vec![SplitFactor::new(4).unwrap()]).unwrap();
    schedule.apply(Primitive::Bind { loop_: parts[0], axis: ThreadAxis::BlockX }).unwrap();
    schedule.apply(Primitive::Bind { loop_: parts[1], axis: ThreadAxis::ThreadX }).unwrap();
    schedule.apply(Primitive::CacheRead { tensor: TensorParamId::new(0), at: nest[1] }).unwrap();
    (compute, schedule)
}

#[test]
fn a_staged_tile_holds_zero_padding_copied_or_other_padding_stored() {
    let inputs = [tensor(&[12], (1..=12).collect::<Vec<i32>>())];
    for padding in [0, -100] {
        let (compute, schedule) = staged_window(padding);

        let kernel = lower(schedule);

        let expected = run_default(&compute, &inputs).unwrap();
        assert_eq!(ints(&run(&kernel, &inputs).unwrap()[0]), ints(&expected[0]));
        let printed = print_kernel(&kernel);
        assert_eq!(printed.contains("<- async"), padding == 0, "{printed}");
    }
}

#[test]
fn a_staged_tile_of_an_empty_input_holds_only_its_padding() {
    let mut compute = ComputeBuilder::new(ComputeName::new("empty"));
    let x = compute.input(TensorName::new("x"), DType::I32, shape(&[0]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(4));
    let read = shifted(&mut compute, x, i, Scalar::int(DType::I32, 5).unwrap());
    compute.output(TensorName::new("out"), read);
    let compute = Arc::new(compute.finish().unwrap());
    let (mut schedule, nest) = open(&compute);
    let parts = schedule.split(nest[0], vec![SplitFactor::new(2).unwrap()]).unwrap();
    schedule.apply(Primitive::Bind { loop_: parts[1], axis: ThreadAxis::ThreadX }).unwrap();
    schedule.apply(Primitive::CacheRead { tensor: x, at: parts[0] }).unwrap();

    let kernel = lower(schedule);

    let out = run(&kernel, &[tensor(&[0], Vec::<i32>::new())]).unwrap();
    assert_eq!(ints(&out[0]), [5, 5, 5, 5]);
    assert!(!print_kernel(&kernel).contains("%x["), "{}", print_kernel(&kernel));
}

#[test]
fn a_tensor_read_with_two_paddings_has_no_one_tile() {
    let mut compute = ComputeBuilder::new(ComputeName::new("two"));
    let x = compute.input(TensorName::new("x"), DType::I32, shape(&[3]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(5));
    let [zero, one] = [0, 1].map(|value| Scalar::int(DType::I32, value).unwrap());
    let (first, second) = (shifted(&mut compute, x, i, zero), shifted(&mut compute, x, i, one));
    let both = TensorExpr::Binary(ptxc_loop::BinaryOp::Add, Box::new(first), Box::new(second));
    compute.output(TensorName::new("out"), both);
    let compute = Arc::new(compute.finish().unwrap());
    let (mut schedule, nest) = open(&compute);

    let staged = schedule.apply(Primitive::CacheRead { tensor: x, at: nest[0] });

    assert_eq!(staged, Err(ScheduleError::CacheLoads(TensorName::new("x"))));
}
