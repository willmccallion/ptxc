//! Running lowered kernels: results against hand-computed values, and
//! scheduled kernels against the default schedule.

use crate::compute_build::{binary, load, shape};
use crate::lower::lower;
use crate::schedule_support::{open, reduction};
use proptest::prelude::Strategy;
use ptxc_core::{DType, Dim, HostTensor, TensorData};
use ptxc_loop::{
    ArgKind, BinaryOp, ComputeBuilder, ComputeDef, ComputeName, Primitive, ReduceOp, RunError,
    Schedule, SplitFactor, TensorExpr, TensorName, ThreadAxis, TieBreak, VarName, run,
};
use std::sync::Arc;

fn tensor(dims: &[u64], data: impl Into<TensorData>) -> HostTensor {
    HostTensor::new(shape(dims), data.into()).unwrap()
}

fn floats(output: &HostTensor) -> Vec<f32> {
    let TensorData::F32(values) = output.data() else { panic!("{:?}", output.dtype()) };
    values.clone()
}

fn ints(output: &HostTensor) -> Vec<i64> {
    match output.data() {
        TensorData::I64(values) => values.clone(),
        TensorData::I32(values) => values.iter().map(|&value| i64::from(value)).collect(),
        other => panic!("{:?}", other.dtype()),
    }
}

/// `out[i, j] = sum over k of x[i, j, k]` on `i32`, whose sums are exact
/// in any order.
fn int_reduction(extents: [u64; 3]) -> Arc<ComputeDef> {
    let [i_extent, j_extent, k_extent] = extents;
    let mut compute = ComputeBuilder::new(ComputeName::new("int_sum"));
    let x = compute.input(TensorName::new("x"), DType::I32, shape(&extents));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(i_extent));
    let (_, j) = compute.spatial(VarName::new("j"), Dim::new(j_extent));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(k_extent));
    let body = Box::new(load(x, &[i, j, k]));
    compute.output(
        TensorName::new("out"),
        TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body },
    );
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

fn run_default(
    compute: &Arc<ComputeDef>,
    inputs: &[HostTensor],
) -> Result<Vec<HostTensor>, RunError> {
    let (schedule, _) = open(compute);
    run(&lower(schedule), inputs)
}

#[test]
fn an_elementwise_kernel_computes_each_element_once() {
    let mut compute = ComputeBuilder::new(ComputeName::new("add"));
    let left = compute.input(TensorName::new("x"), DType::F32, shape(&[5]));
    let right = compute.input(TensorName::new("y"), DType::F32, shape(&[5]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(5));
    compute
        .output(TensorName::new("sum"), binary(BinaryOp::Add, load(left, &[i]), load(right, &[i])));
    let compute = Arc::new(compute.finish().unwrap());
    let (mut schedule, nest) = open(&compute);
    let parts = schedule.split(nest[0], vec![SplitFactor::new(2).unwrap()]).unwrap();
    schedule.apply(Primitive::Bind { loop_: parts[0], axis: ThreadAxis::BlockX }).unwrap();
    schedule.apply(Primitive::Bind { loop_: parts[1], axis: ThreadAxis::ThreadX }).unwrap();
    let inputs = [
        tensor(&[5], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]),
        tensor(&[5], vec![10.0_f32, 20.0, 30.0, 40.0, 50.0]),
    ];

    let outputs = run(&lower(schedule), &inputs).unwrap();

    assert_eq!(floats(&outputs[0]), [11.0, 22.0, 33.0, 44.0, 55.0]);
}

#[test]
fn a_two_dimensional_grid_of_two_dimensional_blocks_covers_every_element() {
    let mut compute = ComputeBuilder::new(ComputeName::new("scale"));
    let x = compute.input(TensorName::new("x"), DType::I32, shape(&[6, 5]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(6));
    let (_, j) = compute.spatial(VarName::new("j"), Dim::new(5));
    let ten = TensorExpr::Const(ptxc_loop::Scalar::int(DType::I32, 10).unwrap());
    compute.output(TensorName::new("y"), binary(BinaryOp::Mul, load(x, &[i, j]), ten));
    let compute = Arc::new(compute.finish().unwrap());
    let (mut schedule, nest) = open(&compute);
    let rows = schedule.split(nest[0], vec![SplitFactor::new(4).unwrap()]).unwrap();
    let columns = schedule.split(nest[1], vec![SplitFactor::new(2).unwrap()]).unwrap();
    for (loop_, axis) in [
        (rows[0], ThreadAxis::BlockY),
        (rows[1], ThreadAxis::ThreadY),
        (columns[0], ThreadAxis::BlockX),
        (columns[1], ThreadAxis::ThreadX),
    ] {
        schedule.apply(Primitive::Bind { loop_, axis }).unwrap();
    }
    let input: Vec<i32> = (0..30).collect();

    let outputs = run(&lower(schedule), &[tensor(&[6, 5], input.clone())]).unwrap();

    let expected: Vec<i64> = input.iter().map(|&value| i64::from(value) * 10).collect();
    assert_eq!(ints(&outputs[0]), expected);
}

#[test]
fn every_axis_of_grid_and_block_is_indexed() {
    let mut compute = ComputeBuilder::new(ComputeName::new("copy"));
    let x = compute.input(TensorName::new("x"), DType::I32, shape(&[4, 3, 4]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(4));
    let (_, j) = compute.spatial(VarName::new("j"), Dim::new(3));
    let (_, k) = compute.spatial(VarName::new("k"), Dim::new(4));
    compute.output(TensorName::new("y"), load(x, &[i, j, k]));
    let compute = Arc::new(compute.finish().unwrap());
    let (mut schedule, nest) = open(&compute);
    let depth = schedule.split(nest[2], vec![SplitFactor::new(2).unwrap()]).unwrap();
    for (loop_, axis) in [
        (nest[0], ThreadAxis::ThreadX),
        (nest[1], ThreadAxis::ThreadY),
        (depth[1], ThreadAxis::ThreadZ),
        (depth[0], ThreadAxis::BlockZ),
    ] {
        schedule.apply(Primitive::Bind { loop_, axis }).unwrap();
    }
    let input: Vec<i32> = (0..48).collect();

    let outputs = run(&lower(schedule), &[tensor(&[4, 3, 4], input.clone())]).unwrap();

    assert_eq!(ints(&outputs[0]), input.iter().map(|&value| i64::from(value)).collect::<Vec<_>>());
}

#[test]
fn a_sum_adds_along_its_axis() {
    let compute = reduction([2, 1, 3]);
    let input = tensor(&[2, 1, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.5]);

    let outputs = run_default(&compute, &[input]).unwrap();

    assert_eq!(floats(&outputs[0]), [6.0, 15.5]);
}

/// The running sum of `x: i32[4]`, as a sum over `k` masked to `k <= i`.
fn running_sum() -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("running_sum"));
    let x = compute.input(TensorName::new("x"), DType::I32, shape(&[4]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(4));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(4));
    let index = |index| TensorExpr::Index { index, dtype: DType::I64 };
    let zero = TensorExpr::Const(ptxc_loop::Scalar::zero(DType::I32));
    let masked = TensorExpr::Select {
        condition: Box::new(binary(BinaryOp::LessOrEqual, index(k), index(i))),
        then: Box::new(load(x, &[k])),
        otherwise: Box::new(zero),
    };
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body: Box::new(masked) };
    compute.output(TensorName::new("y"), sum);
    Arc::new(compute.finish().unwrap())
}

#[test]
fn an_index_read_as_a_value_masks_a_running_sum() {
    let compute = running_sum();
    let input = tensor(&[4], vec![1_i32, 2, 3, 4]);

    let outputs = run_default(&compute, &[input]).unwrap();

    assert_eq!(ints(&outputs[0]), [1, 3, 6, 10]);
    ptxc_testkit::golden::check("compute/running_sum.txt", &ptxc_loop::print_compute(&compute))
        .unwrap();
}

#[test]
fn an_index_value_may_read_a_gathered_index() {
    let mut compute = ComputeBuilder::new(ComputeName::new("echo"));
    let positions = compute.input(TensorName::new("idx"), DType::I32, shape(&[3]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(3));
    let gathered = compute.loaded_index(VarName::new("j"), positions, vec![i]);
    let doubled = compute.exprs().add(gathered, gathered);
    compute.output(TensorName::new("y"), TensorExpr::Index { index: doubled, dtype: DType::I64 });
    let compute = Arc::new(compute.finish().unwrap());

    let outputs = run_default(&compute, &[tensor(&[3], vec![7_i32, -2, 0])]).unwrap();

    assert_eq!(ints(&outputs[0]), [14, -4, 0]);
}

#[test]
fn argmax_finds_the_first_or_last_maximum_and_the_first_nan() {
    let rows = vec![
        1.0_f32,
        3.0,
        2.0,
        3.0,
        0.0,
        f32::NAN,
        1.0,
        f32::NAN,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let input = || tensor(&[3, 4], rows.clone());

    let first = run_default(&argmax(3, 4, TieBreak::First), &[input()]).unwrap();
    let last = run_default(&argmax(3, 4, TieBreak::Last), &[input()]).unwrap();

    assert_eq!(ints(&first[0]), [1, 1, 0]);
    assert_eq!(ints(&last[0]), [3, 1, 3]);
}

#[test]
fn integer_division_by_zero_is_an_error() {
    let mut compute = ComputeBuilder::new(ComputeName::new("divide"));
    let left = compute.input(TensorName::new("x"), DType::I32, shape(&[2]));
    let right = compute.input(TensorName::new("y"), DType::I32, shape(&[2]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(2));
    compute
        .output(TensorName::new("q"), binary(BinaryOp::Div, load(left, &[i]), load(right, &[i])));
    let compute = Arc::new(compute.finish().unwrap());

    let error =
        run_default(&compute, &[tensor(&[2], vec![4_i32, 5]), tensor(&[2], vec![2_i32, 0])]);

    assert_eq!(error.err(), Some(RunError::DivisionByZero));
}

#[test]
fn a_gather_index_outside_its_tensor_is_an_error() {
    let mut compute = ComputeBuilder::new(ComputeName::new("gather"));
    let data = compute.input(TensorName::new("data"), DType::F32, shape(&[4]));
    let indices = compute.input(TensorName::new("indices"), DType::I64, shape(&[2]));
    let (_, n) = compute.spatial(VarName::new("n"), Dim::new(2));
    let row = compute.loaded_index(VarName::new("row"), indices, vec![n]);
    compute.output(TensorName::new("out"), load(data, &[row]));
    let compute = Arc::new(compute.finish().unwrap());
    let data = || tensor(&[4], vec![1.0_f32, 2.0, 3.0, 4.0]);

    let good = run_default(&compute, &[data(), tensor(&[2], vec![3_i64, 0])]).unwrap();
    let bad = run_default(&compute, &[data(), tensor(&[2], vec![1_i64, 9])]);

    assert_eq!(floats(&good[0]), [4.0, 1.0]);
    assert_eq!(
        bad.err(),
        Some(RunError::OutOfBounds { buffer: TensorName::new("data"), index: 9 })
    );
}

#[test]
fn inputs_must_match_the_kernel() {
    let compute = reduction([2, 1, 3]);

    assert_eq!(
        run_default(&compute, &[]).err(),
        Some(RunError::InputCount { expected: 1, actual: 0 })
    );
    let wrong = tensor(&[6], vec![0_i32; 6]);
    assert_eq!(
        run_default(&compute, &[wrong]).err(),
        Some(RunError::Input { index: 0, dtype: DType::F32, elements: 6 })
    );
}

/// The axes random schedules bind loops to.
const AXES: [ThreadAxis; 4] =
    [ThreadAxis::BlockX, ThreadAxis::BlockY, ThreadAxis::ThreadX, ThreadAxis::ThreadY];

/// A random schedule's steps: splits, fuses and reorders by nest position,
/// then bindings of nest positions to axes, tried in order.
fn scheduled(
    compute: &Arc<ComputeDef>,
    steps: &[crate::schedule_properties::Step],
    binds: &[(usize, usize)],
) -> Option<Schedule<ptxc_loop::Validated>> {
    let (mut schedule, _) = open(compute);
    for step in steps {
        let nest = schedule.loops().nest().to_vec();
        if let Some(primitive) = crate::schedule_properties::primitive(step, &nest) {
            let _rejected_or_applied = schedule.apply(primitive);
        }
    }
    for &(position, axis) in binds {
        let Some(&loop_) = schedule.loops().nest().get(position) else { continue };
        let _rejected_or_applied =
            schedule.apply(Primitive::Bind { loop_, axis: AXES[axis % AXES.len()] });
    }
    schedule.validate(&ptxc_target::preset(ptxc_target::SmVersion::SM_89).unwrap()).ok()
}

proptest::proptest! {
    #![proptest_config(crate::cases(256))]

    /// Any schedule of an integer sum gives the default schedule's result
    /// exactly, cross-thread trees included.
    #[test]
    fn every_schedule_of_an_integer_sum_agrees(
        extents in [1_u64..=5, 1_u64..=4, 1_u64..=9],
        steps in proptest::collection::vec(crate::schedule_properties::step(), 0..6),
        binds in proptest::collection::vec((0_usize..6, 0_usize..4), 0..4),
        data in proptest::collection::vec(-1000_i32..=1000, 180),
    ) {
        let compute = int_reduction(extents);
        let count = usize::try_from(extents.iter().product::<u64>()).unwrap();
        let input = tensor(&extents, data[..count].to_vec());
        let expected = run_default(&compute, std::slice::from_ref(&input)).unwrap();
        if let Some(validated) = scheduled(&compute, &steps, &binds) {
            let actual = run(&validated.lower(), &[input]).unwrap();
            proptest::prop_assert_eq!(ints(&actual[0]), ints(&expected[0]));
        }
    }

    /// Any schedule of an arg-reduction finds the same position, ties and
    /// NaNs included.
    #[test]
    fn every_schedule_of_an_argmax_agrees(
        columns in 1_u64..=20,
        steps in proptest::collection::vec(crate::schedule_properties::step(), 0..6),
        binds in proptest::collection::vec((0_usize..6, 0_usize..4), 0..4),
        values in proptest::collection::vec(proptest::prelude::prop_oneof![
            (-3_i16..=3).prop_map(f32::from),
            proptest::prelude::Just(f32::NAN),
            proptest::prelude::Just(f32::NEG_INFINITY),
        ], 60),
        last in proptest::prelude::any::<bool>(),
    ) {
        let tie = if last { TieBreak::Last } else { TieBreak::First };
        let compute = argmax(3, columns, tie);
        let count = usize::try_from(3 * columns).unwrap();
        let input = tensor(&[3, columns], values[..count].to_vec());
        let expected = run_default(&compute, std::slice::from_ref(&input)).unwrap();
        if let Some(validated) = scheduled(&compute, &steps, &binds) {
            let actual = run(&validated.lower(), &[input]).unwrap();
            proptest::prop_assert_eq!(ints(&actual[0]), ints(&expected[0]));
        }
    }
}
