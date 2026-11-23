//! Lowering validated schedules to kernels, checked by the verifier and
//! reviewed as golden text.

use crate::compute_build::{binary, load, shape};
use crate::schedule_support::{open, reduction};
use ptxc_core::{DType, Dim};
use ptxc_loop::{
    ArgKind, BinaryOp, ComputeBuilder, ComputeDef, ComputeName, LoopId, LoweredKernel, Open,
    Primitive, Schedule, SplitFactor, TensorExpr, TensorName, ThreadAxis, TieBreak, VarName,
    print_kernel, verify,
};
use ptxc_target::{SmVersion, preset};
use ptxc_testkit::golden;
use std::sync::Arc;

pub(crate) fn lower(schedule: Schedule<Open>) -> LoweredKernel {
    let kernel = schedule.validate(&preset(SmVersion::SM_89).unwrap()).unwrap().lower();
    verify(&kernel).unwrap_or_else(|error| panic!("{error}\n{}", print_kernel(&kernel)));
    kernel
}

fn split(schedule: &mut Schedule<Open>, loop_: LoopId, factor: u64) -> Vec<LoopId> {
    schedule.split(loop_, vec![SplitFactor::new(factor).unwrap()]).unwrap()
}

fn bind(schedule: &mut Schedule<Open>, loop_: LoopId, axis: ThreadAxis) {
    schedule.apply(Primitive::Bind { loop_, axis }).unwrap();
}

fn add(length: u64) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("add"));
    let left = compute.input(TensorName::new("x"), DType::F32, shape(&[length]));
    let right = compute.input(TensorName::new("y"), DType::F32, shape(&[length]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(length));
    compute
        .output(TensorName::new("sum"), binary(BinaryOp::Add, load(left, &[i]), load(right, &[i])));
    Arc::new(compute.finish().unwrap())
}

fn argmax(rows: u64, columns: u64, tie: TieBreak) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("argmax"));
    let x = compute.input(TensorName::new("x"), DType::F32, shape(&[rows, columns]));
    let (_, row) = compute.spatial(VarName::new("row"), Dim::new(rows));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(columns));
    let body = Box::new(load(x, &[row, k]));
    compute.output(
        TensorName::new("position"),
        TensorExpr::ArgReduce { kind: ArgKind::Max, tie, axis: k_axis, body },
    );
    Arc::new(compute.finish().unwrap())
}

fn gather() -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("gather"));
    let data = compute.input(TensorName::new("data"), DType::F32, shape(&[10, 4]));
    let indices = compute.input(TensorName::new("indices"), DType::I64, shape(&[3]));
    let (_, n) = compute.spatial(VarName::new("n"), Dim::new(3));
    let (_, c) = compute.spatial(VarName::new("c"), Dim::new(4));
    let row = compute.loaded_index(VarName::new("row"), indices, vec![n]);
    let ten = compute.exprs().constant(10);
    let wrapped = compute.exprs().floor_mod(row, ten);
    compute.output(TensorName::new("out"), load(data, &[wrapped, c]));
    Arc::new(compute.finish().unwrap())
}

#[test]
fn an_elementwise_kernel_binds_its_loops_and_guards_the_tail() {
    let compute = add(1000);
    let (mut schedule, nest) = open(&compute);
    let parts = split(&mut schedule, nest[0], 256);
    bind(&mut schedule, parts[0], ThreadAxis::BlockX);
    bind(&mut schedule, parts[1], ThreadAxis::ThreadX);

    let kernel = lower(schedule);

    golden::check("lower/add.txt", &print_kernel(&kernel)).unwrap();
}

#[test]
fn a_tiled_reduction_accumulates_per_inner_point() {
    let compute = reduction([64, 30, 32]);
    let (mut schedule, nest) = open(&compute);
    let i = split(&mut schedule, nest[0], 16);
    let j = split(&mut schedule, nest[1], 8);
    let k = split(&mut schedule, nest[2], 4);
    schedule.apply(Primitive::Reorder { order: vec![i[0], j[0], k[0], i[1], j[1], k[1]] }).unwrap();
    bind(&mut schedule, i[0], ThreadAxis::BlockX);
    bind(&mut schedule, j[0], ThreadAxis::BlockY);
    schedule.apply(Primitive::Unroll { loop_: k[1] }).unwrap();

    let kernel = lower(schedule);

    golden::check("lower/tiled_reduction.txt", &print_kernel(&kernel)).unwrap();
}

#[test]
fn a_reduction_shared_between_threads_combines_in_a_tree() {
    let compute = reduction([4, 3, 100]);
    let (mut schedule, nest) = open(&compute);
    let k = split(&mut schedule, nest[2], 32);
    bind(&mut schedule, nest[0], ThreadAxis::BlockX);
    bind(&mut schedule, k[1], ThreadAxis::ThreadX);

    let kernel = lower(schedule);

    assert_eq!(kernel.smem().get(), 32 * 4);
    golden::check("lower/cross_thread_sum.txt", &print_kernel(&kernel)).unwrap();
}

#[test]
fn an_arg_reduction_tracks_its_position_in_and_across_threads() {
    let (mut serial, rows) = open(&argmax(5, 7, TieBreak::First));
    bind(&mut serial, rows[0], ThreadAxis::BlockX);
    let compute = argmax(5, 40, TieBreak::Last);
    let (mut shared, nest) = open(&compute);
    let k = split(&mut shared, nest[1], 16);
    bind(&mut shared, k[1], ThreadAxis::ThreadX);

    golden::check("lower/argmax.txt", &print_kernel(&lower(serial))).unwrap();
    golden::check("lower/cross_thread_argmax.txt", &print_kernel(&lower(shared))).unwrap();
}

#[test]
fn a_gather_binds_its_index_before_the_load() {
    let (schedule, _) = open(&gather());

    golden::check("lower/gather.txt", &print_kernel(&lower(schedule))).unwrap();
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(256))]

    /// Whatever splits, fuses and reorders shape a schedule, it lowers to a
    /// kernel the verifier accepts.
    #[test]
    fn every_schedule_lowers_to_a_verified_kernel(
        extents in [1_u64..=6, 1_u64..=6, 1_u64..=5],
        steps in proptest::collection::vec(crate::schedule_properties::step(), 0..8),
    ) {
        let compute = reduction(extents);
        let (mut schedule, _) = open(&compute);
        for step in &steps {
            let nest = schedule.loops().nest().to_vec();
            if let Some(primitive) = crate::schedule_properties::primitive(step, &nest) {
                let _rejected_or_applied = schedule.apply(primitive);
            }
        }

        let _kernel = lower(schedule);
    }
}

/// `any[i] = max over k of (x[i, k] > 0)`: a reduction of booleans.
fn any_positive() -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("any_positive"));
    let x = compute.input(TensorName::new("x"), DType::F32, shape(&[4, 6]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(4));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(6));
    let zero = TensorExpr::Const(ptxc_loop::Scalar::float(DType::F32, 0.0).unwrap());
    let positive = binary(BinaryOp::Greater, load(x, &[i, k]), zero);
    let body = Box::new(positive);
    compute.output(
        TensorName::new("any"),
        TensorExpr::Reduce { op: ptxc_loop::ReduceOp::Max, axes: vec![k_axis], body },
    );
    Arc::new(compute.finish().unwrap())
}

/// `out[i] = sum over k of data[indices[k], i]`: a gather inside a reduction.
fn gathered_sum() -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("gathered_sum"));
    let data = compute.input(TensorName::new("data"), DType::F32, shape(&[10, 3]));
    let indices = compute.input(TensorName::new("indices"), DType::I32, shape(&[5]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(3));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(5));
    let row = compute.loaded_index(VarName::new("row"), indices, vec![k]);
    let body = Box::new(load(data, &[row, i]));
    compute.output(
        TensorName::new("out"),
        TensorExpr::Reduce { op: ptxc_loop::ReduceOp::Sum, axes: vec![k_axis], body },
    );
    Arc::new(compute.finish().unwrap())
}

#[test]
fn a_reduction_of_booleans_keeps_a_boolean_accumulator() {
    let (schedule, _) = open(&any_positive());

    golden::check("lower/any_positive.txt", &print_kernel(&lower(schedule))).unwrap();
}

#[test]
fn a_gather_inside_a_reduction_is_bound_inside_its_loop() {
    let (schedule, _) = open(&gathered_sum());

    golden::check("lower/gathered_sum.txt", &print_kernel(&lower(schedule))).unwrap();
}

#[test]
fn a_reduction_on_the_y_axis_steps_by_the_x_extent() {
    let compute = reduction([8, 2, 64]);
    let (mut schedule, nest) = open(&compute);
    let k = split(&mut schedule, nest[2], 16);
    bind(&mut schedule, nest[0], ThreadAxis::ThreadX);
    bind(&mut schedule, k[1], ThreadAxis::ThreadY);

    let kernel = lower(schedule);

    golden::check("lower/cross_thread_rows.txt", &print_kernel(&kernel)).unwrap();
}

/// `largest[i] = max over k of x[i, k] * 2`: a reduction whose body is
/// arithmetic, so its accumulator is a float combined with `max`.
fn largest_double() -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("largest_double"));
    let x = compute.input(TensorName::new("x"), DType::F32, shape(&[3, 5]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(3));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(5));
    let two = TensorExpr::Const(ptxc_loop::Scalar::float(DType::F32, 2.0).unwrap());
    let body = Box::new(binary(BinaryOp::Mul, load(x, &[i, k]), two));
    compute.output(
        TensorName::new("largest"),
        TensorExpr::Reduce { op: ptxc_loop::ReduceOp::Max, axes: vec![k_axis], body },
    );
    Arc::new(compute.finish().unwrap())
}

#[test]
fn a_reduction_of_arithmetic_keeps_its_number_type() {
    let (schedule, _) = open(&largest_double());

    golden::check("lower/largest_double.txt", &print_kernel(&lower(schedule))).unwrap();
}
