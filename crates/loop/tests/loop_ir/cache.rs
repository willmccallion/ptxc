//! Inputs staged in shared memory with `cache_read`: the kernels a stage
//! lowers to, and that staging never changes a result.

use crate::compute_build::{binary, load, shape};
use crate::interp::{ints, run_default, shaped, target, tensor};
use crate::lower::lower;
use crate::schedule_support::open;
use proptest::prelude::*;
use ptxc_core::{DType, Dim, HostTensor, Idx};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeDef, ComputeName, LoopId, Open, Primitive, ReduceOp, Schedule,
    SplitFactor, TensorExpr, TensorName, TensorParamId, ThreadAxis, VarName, print_kernel,
    print_schedule, run,
};
use ptxc_testkit::golden;
use std::sync::Arc;

/// `out[i, j] = sum over k of a[i, k] * b[k, j]` on `i32`, exact in any
/// order.
pub(crate) fn matmul([rows, columns, depth]: [u64; 3]) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("matmul"));
    let left = compute.input(TensorName::new("a"), DType::I32, shape(&[rows, depth]));
    let right = compute.input(TensorName::new("b"), DType::I32, shape(&[depth, columns]));
    let (_, row) = compute.spatial(VarName::new("i"), Dim::new(rows));
    let (_, column) = compute.spatial(VarName::new("j"), Dim::new(columns));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(depth));
    let product = binary(BinaryOp::Mul, load(left, &[row, k]), load(right, &[k, column]));
    let body = Box::new(product);
    compute.output(
        TensorName::new("out"),
        TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body },
    );
    Arc::new(compute.finish().unwrap())
}

/// `matmul`'s first operand, `a`.
pub(crate) fn a() -> TensorParamId {
    TensorParamId::new(0)
}

/// `matmul`'s second operand, `b`.
pub(crate) fn b() -> TensorParamId {
    TensorParamId::new(1)
}

/// Blocks of `tile` by `tile` threads, one output each, with the reduction
/// split by `depth`: the nest `io, jo, ii, ji, ko, ki`, and its `ko` and
/// `ki`.
pub(crate) fn tiled(
    compute: &Arc<ComputeDef>,
    [tile, depth]: [u64; 2],
) -> (Schedule<Open>, [LoopId; 2]) {
    let (mut schedule, nest) = open(compute);
    let by = |factor| vec![SplitFactor::new(factor).unwrap()];
    let i = schedule.split(nest[0], by(tile)).unwrap();
    let j = schedule.split(nest[1], by(tile)).unwrap();
    let k = schedule.split(nest[2], by(depth)).unwrap();
    schedule.apply(Primitive::Reorder { order: vec![i[0], j[0], i[1], j[1]] }).unwrap();
    let binds = [
        (i[0], ThreadAxis::BlockY),
        (j[0], ThreadAxis::BlockX),
        (i[1], ThreadAxis::ThreadY),
        (j[1], ThreadAxis::ThreadX),
    ];
    for (loop_, axis) in binds {
        schedule.apply(Primitive::Bind { loop_, axis }).unwrap();
    }
    (schedule, [k[0], k[1]])
}

pub(crate) fn stage(schedule: &mut Schedule<Open>, tensor: TensorParamId, at: LoopId) {
    schedule.apply(Primitive::CacheRead { tensor, at }).unwrap();
}

/// Inputs of `matmul(extents)` counting up from distinct starts, so a read
/// of the wrong element shows.
pub(crate) fn operands([m, n, k]: [u64; 3]) -> [HostTensor; 2] {
    let values = |count: u64, start: i32| -> Vec<i32> {
        (0..i32::try_from(count).unwrap()).map(|value| start + value).collect()
    };
    [tensor(&[m, k], values(m * k, 1)), tensor(&[k, n], values(k * n, -40))]
}

pub(crate) fn agrees_with_default(
    compute: &Arc<ComputeDef>,
    schedule: Schedule<Open>,
    inputs: &[HostTensor],
) {
    let expected = run_default(compute, inputs).unwrap();
    let actual = run(&lower(schedule), inputs).unwrap();
    assert_eq!(ints(&actual[0]), ints(&expected[0]));
}

#[test]
fn a_tiled_matmul_stages_both_operands_at_the_outer_reduction_loop() {
    let compute = matmul([8, 8, 8]);
    let (mut schedule, [ko, _]) = tiled(&compute, [4, 4]);
    stage(&mut schedule, a(), ko);
    stage(&mut schedule, b(), ko);

    let kernel = lower(schedule);

    golden::check("lower/staged_matmul.txt", &print_kernel(&kernel)).unwrap();
    assert_eq!(kernel.smem().get(), 2 * 4 * 4 * 4);
}

#[test]
fn a_tile_larger_than_its_block_fills_in_rounds() {
    let extents = [37, 45, 70];
    let compute = matmul(extents);
    let (mut schedule, [ko, _]) = tiled(&compute, [8, 16]);
    stage(&mut schedule, a(), ko);
    stage(&mut schedule, b(), ko);

    let kernel = lower(schedule.clone());

    golden::check("lower/staged_matmul_rounds.txt", &print_kernel(&kernel)).unwrap();
    agrees_with_default(&compute, schedule, &operands(extents));
}

#[test]
fn an_update_reading_only_tiles_is_guarded_only_on_its_reduction() {
    let compute = matmul([37, 45, 70]);
    let (unstaged, [ko, _]) = tiled(&compute, [8, 16]);
    let mut staged = unstaged.clone();
    stage(&mut staged, a(), ko);
    stage(&mut staged, b(), ko);

    let [unstaged, staged] = [unstaged, staged].map(|schedule| print_kernel(&lower(schedule)));

    // The update and the store each check the row and column bounds where
    // the update reads the inputs; reading tiles, only the store does.
    let guards = |text: &str, bound: &str| text.matches(&format!(" < {bound} {{")).count();
    assert_eq!([guards(&unstaged, "37"), guards(&unstaged, "45")], [2, 2], "{unstaged}");
    assert_eq!([guards(&staged, "37"), guards(&staged, "45")], [1, 1], "{staged}");
    assert_eq!(guards(&staged, "70"), 1, "{staged}");
}

#[test]
fn a_staged_matmul_agrees_where_its_tiles_overhang_the_inputs() {
    let extents = [6, 7, 9];
    let compute = matmul(extents);
    let (mut schedule, [ko, _]) = tiled(&compute, [4, 4]);
    stage(&mut schedule, a(), ko);
    stage(&mut schedule, b(), ko);

    agrees_with_default(&compute, schedule, &operands(extents));
}

#[test]
fn a_stage_at_the_inner_reduction_loop_holds_one_column_per_thread_row() {
    let extents = [5, 6, 7];
    let compute = matmul(extents);
    let (mut schedule, [_, ki]) = tiled(&compute, [3, 3]);
    stage(&mut schedule, a(), ki);

    let kernel = lower(schedule.clone());

    assert_eq!(kernel.smem().get(), 3 * 4);
    agrees_with_default(&compute, schedule, &operands(extents));
}

#[test]
fn a_stage_in_a_serial_nest_is_filled_by_the_one_thread() {
    let extents = [3, 4, 5];
    let compute = matmul(extents);
    let (mut schedule, nest) = open(&compute);
    stage(&mut schedule, b(), nest[0]);

    let kernel = lower(schedule.clone());

    assert_eq!(kernel.smem().get(), 5 * 4 * 4);
    agrees_with_default(&compute, schedule, &operands(extents));
}

#[test]
fn a_tile_whose_origin_falls_below_its_input_past_the_last_point_is_guarded() {
    let mut compute = ComputeBuilder::new(ComputeName::new("reverse"));
    let x = compute.input(TensorName::new("x"), DType::I32, shape(&[3]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(3));
    let exprs = compute.exprs();
    let (minus_one, last) = (exprs.constant(-1), exprs.constant(2));
    let negated = exprs.mul(i, minus_one);
    let reversed = exprs.add(negated, last);
    compute.output(TensorName::new("out"), load(x, &[reversed]));
    let compute = Arc::new(compute.finish().unwrap());
    let (mut schedule, nest) = open(&compute);
    let parts = schedule.split(nest[0], vec![SplitFactor::new(2).unwrap()]).unwrap();
    stage(&mut schedule, x, parts[1]);

    agrees_with_default(&compute, schedule, &[tensor(&[3], vec![7_i32, 8, 9])]);
}

#[test]
fn a_staged_schedule_prints_its_stages_and_replays_them() {
    let compute = matmul([8, 8, 8]);
    let (mut schedule, [ko, ki]) = tiled(&compute, [4, 4]);
    stage(&mut schedule, a(), ko);
    stage(&mut schedule, b(), ki);

    let replayed = Schedule::replay(Arc::clone(&compute), schedule.trace()).unwrap();

    let printed = print_schedule(&schedule);
    assert!(printed.contains("  cache_read a %k.0\n  cache_read b %k.1\n"), "{printed}");
    assert_eq!(print_schedule(&replayed), printed);
    assert_eq!(replayed.loops().caches(), schedule.loops().caches());
}

proptest! {
    #![proptest_config(crate::cases(256))]

    /// Staging either operand of a tiled matmul at either reduction loop
    /// always validates and never changes the result.
    #[test]
    fn staging_a_tiled_matmul_never_changes_its_result(
        extents in [1_u64..=9, 1_u64..=9, 1_u64..=9],
        tile in 1_u64..=4,
        depth in 1_u64..=6,
        stages in prop::collection::vec((0_usize..2, 0_usize..2), 0..=2),
    ) {
        let compute = matmul(extents);
        let (mut schedule, reduction_loops) = tiled(&compute, [tile, depth]);
        for &(tensor, at) in &stages {
            let tensor = TensorParamId::new(tensor);
            if !schedule.loops().caches().iter().any(|cache| cache.tensor == tensor) {
                stage(&mut schedule, tensor, reduction_loops[at]);
            }
        }
        agrees_with_default(&compute, schedule, &operands(extents));
    }

    /// Any schedule with stages that validates computes the default
    /// schedule's result, with no race, divergent barrier or unset read.
    #[test]
    fn every_valid_staged_schedule_agrees(
        extents in [1_u64..=6, 1_u64..=6, 1_u64..=6],
        steps in prop::collection::vec(crate::schedule_properties::step(), 0..6),
        binds in prop::collection::vec((0_usize..6, 0_usize..4), 0..4),
        stages in prop::collection::vec((0_usize..2, 0_usize..8), 1..=2),
    ) {
        let compute = matmul(extents);
        let mut schedule = shaped(&compute, &steps, &binds);
        for &(tensor, position) in &stages {
            let nest = schedule.loops().nest();
            let at = nest[position % nest.len()];
            let tensor = TensorParamId::new(tensor);
            let _rejected_or_applied = schedule.apply(Primitive::CacheRead { tensor, at });
        }
        if let Ok(validated) = schedule.validate(&target()) {
            let inputs = operands(extents);
            let expected = run_default(&compute, &inputs).unwrap();
            let actual = run(&validated.lower(), &inputs).unwrap();
            prop_assert_eq!(ints(&actual[0]), ints(&expected[0]));
        }
    }
}
