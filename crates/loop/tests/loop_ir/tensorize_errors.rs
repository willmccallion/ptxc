//! Where `tensorize` refuses a tile: at apply, for the loops' extents,
//! kinds and the lanes' axis, and at validation, for where the tile ends up
//! and what the definition computes.

use crate::compute_build::shape;
use crate::interp::target;
use crate::schedule_support::open;
use crate::tensorize::{instruction, product, product_reading, tensorized};
use ptxc_core::{DType, Dim, Idx};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeDef, ComputeName, LoopId, Open, Primitive, ReduceOp, Schedule,
    ScheduleError, SplitFactor, TensorExpr, TensorName, TensorParamId, ThreadAxis, TileOrder,
    VarName,
};
use ptxc_testkit::tensorized::{Loads, ONE_TILE};
use std::sync::Arc;

fn split(schedule: &mut Schedule<Open>, loop_: LoopId, factor: u64) -> Vec<LoopId> {
    schedule.split(loop_, vec![SplitFactor::new(factor).unwrap()]).unwrap()
}

fn tensorize(schedule: &mut Schedule<Open>, [m, n, k]: [LoopId; 3]) -> Result<(), ScheduleError> {
    schedule.apply(Primitive::Tensorize { m, n, k, instruction: instruction() })
}

fn validation_error(schedule: Schedule<Open>) -> ScheduleError {
    schedule.validate(&target()).unwrap_err()
}

/// A 16 by 8 by 32 product split into its tile and a reduction step, the
/// tile innermost: its loops, rows, columns, step and depth.
fn tile_and_step(schedule: &mut Schedule<Open>, nest: &[LoopId]) -> [LoopId; 4] {
    let depth = split(schedule, nest[2], 16);
    [nest[0], nest[1], depth[0], depth[1]]
}

#[test]
fn a_tile_takes_loops_of_the_mmas_extents_and_kinds() {
    let compute = product([8, 8, 32], TileOrder::RowMajor);
    let (mut short, nest) = open(&compute);
    let [rows, columns, _, depth] = tile_and_step(&mut short, &nest);
    let mut swapped = open(&product([16, 8, 32], TileOrder::RowMajor)).0;
    let swapped_nest = swapped.loops().nest().to_vec();
    let [rows_16, columns_8, _, depth_16] = tile_and_step(&mut swapped, &swapped_nest);

    let too_short = tensorize(&mut short, [rows, columns, depth]);
    let wrong_kind = tensorize(&mut swapped, [depth_16, columns_8, rows_16]);

    let (loop_, expected, actual) = (VarName::new("i"), Dim::new(16), Dim::new(8));
    assert_eq!(too_short, Err(ScheduleError::TensorizeExtent { loop_, expected, actual }));
    assert_eq!(wrong_kind, Err(ScheduleError::TensorizeKind(VarName::new("k.1"))));
}

#[test]
fn a_nest_is_tensorized_once_and_its_lanes_keep_threadidx_x() {
    let compute = product([16, 8, 32], TileOrder::RowMajor);
    let (mut schedule, nest) = open(&compute);
    let [rows, columns, step, depth] = tile_and_step(&mut schedule, &nest);
    schedule.apply(Primitive::Reorder { order: vec![step, rows, columns, depth] }).unwrap();
    tensorize(&mut schedule, [rows, columns, depth]).unwrap();
    let (mut bound_first, bound_nest) = open(&compute);
    let [bound_rows, bound_columns, bound_step, bound_depth] =
        tile_and_step(&mut bound_first, &bound_nest);
    bound_first.apply(Primitive::Bind { loop_: bound_step, axis: ThreadAxis::ThreadX }).unwrap();

    let again = tensorize(&mut schedule, [rows, columns, depth]);
    let lanes_bound = schedule.apply(Primitive::Bind { loop_: step, axis: ThreadAxis::ThreadX });
    let after_binding = tensorize(&mut bound_first, [bound_rows, bound_columns, bound_depth]);

    assert_eq!(again, Err(ScheduleError::AlreadyTensorized));
    assert_eq!(lanes_bound, Err(ScheduleError::AxisTaken(ThreadAxis::ThreadX)));
    assert_eq!(after_binding, Err(ScheduleError::AxisTaken(ThreadAxis::ThreadX)));
}

#[test]
fn a_tile_must_be_innermost_inside_a_reduction_step() {
    let compute = product([16, 8, 32], TileOrder::RowMajor);
    let mut moved = tensorized(&compute, ONE_TILE);
    let nest = moved.loops().nest().to_vec();
    let last = nest.len() - 1;
    moved.apply(Primitive::Reorder { order: vec![nest[last], nest[last - 1]] }).unwrap();
    let (mut stepless, stepless_nest) = open(&product([16, 8, 16], TileOrder::RowMajor));
    let order = vec![stepless_nest[0], stepless_nest[1], stepless_nest[2]];
    stepless.apply(Primitive::Reorder { order }).unwrap();
    tensorize(&mut stepless, [stepless_nest[0], stepless_nest[1], stepless_nest[2]]).unwrap();

    assert_eq!(validation_error(moved), ScheduleError::TensorizeNotInnermost);
    assert_eq!(validation_error(stepless), ScheduleError::TensorizeNoStep);
}

#[test]
fn a_tile_computes_a_product_of_staged_operands_of_the_mmas_types() {
    let mut unstaged = open(&product([16, 8, 32], TileOrder::RowMajor)).0;
    stage_and_tensorize(&mut unstaged, [Some(Stage::Step), None]);
    let mut staged_inside = open(&product([16, 8, 32], TileOrder::RowMajor)).0;
    stage_and_tensorize(&mut staged_inside, [Some(Stage::Rows), Some(Stage::Step)]);

    assert_eq!(validation_error(unstaged), ScheduleError::TensorizeUnstaged(TensorName::new("b")));
    assert_eq!(
        validation_error(staged_inside),
        ScheduleError::TensorizeUnstaged(TensorName::new("a"))
    );
    for dtype in [DType::F32, DType::F16] {
        let compute = unwidened_product(dtype);
        assert_eq!(
            validation_error(tensorized(&compute, ONE_TILE)),
            ScheduleError::TensorizeNotProduct
        );
    }
}

#[test]
fn an_operands_tile_has_adjacent_elements_and_reads_only_its_own_loops() {
    let shifted: Box<Loads> = Box::new(|compute, [a, b], [i, j, k]| {
        [(a, vec![i, compute.exprs().add(k, j)]), (b, vec![k, j])]
    });
    let sparse: Box<Loads> = Box::new(|compute, [a, b], [i, j, k]| {
        let two = compute.exprs().constant(2);
        [(a, vec![compute.exprs().mul(i, two), compute.exprs().mul(k, two)]), (b, vec![k, j])]
    });
    let products = [
        product_reading([&[16, 39], &[32, 8]], [16, 8, 32], shifted),
        product_reading([&[31, 63], &[32, 8]], [16, 8, 32], sparse),
    ];

    for compute in products {
        assert_eq!(
            validation_error(tensorized(&compute, ONE_TILE)),
            ScheduleError::TensorizeIndex(TensorName::new("a"))
        );
    }
}

/// Where a test stages an operand: at the reduction step, outside the
/// tile, or at the tile's rows loop, inside it.
#[derive(Clone, Copy)]
enum Stage {
    Step,
    Rows,
}

/// `schedule`'s 16 by 8 by 32 product tensorized inside one reduction
/// step, `a` and `b` staged where `stages` says, if at all.
fn stage_and_tensorize(schedule: &mut Schedule<Open>, stages: [Option<Stage>; 2]) {
    let nest = schedule.loops().nest().to_vec();
    let [rows, columns, step, depth] = tile_and_step(schedule, &nest);
    schedule.apply(Primitive::Reorder { order: vec![step, rows, columns, depth] }).unwrap();
    for (tensor, stage) in stages.into_iter().enumerate() {
        let at = match stage {
            Some(Stage::Step) => step,
            Some(Stage::Rows) => rows,
            None => continue,
        };
        schedule.apply(Primitive::CacheRead { tensor: TensorParamId::new(tensor), at }).unwrap();
    }
    tensorize(schedule, [rows, columns, depth]).unwrap();
}

/// `out[i, j] = sum over k of a[i, k] * b[k, j]` on `dtype` operands
/// multiplied as they are, which the f16 to f32 mma does not compute.
fn unwidened_product(dtype: DType) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("unwidened"));
    let lhs = compute.input(TensorName::new("a"), dtype, shape(&[16, 32]));
    let rhs = compute.input(TensorName::new("b"), dtype, shape(&[32, 8]));
    let (_, row) = compute.spatial(VarName::new("i"), Dim::new(16));
    let (_, column) = compute.spatial(VarName::new("j"), Dim::new(8));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(32));
    let load = |tensor, index| Box::new(TensorExpr::Load { tensor, index });
    let body = Box::new(TensorExpr::Binary(
        BinaryOp::Mul,
        load(lhs, vec![row, k]),
        load(rhs, vec![k, column]),
    ));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body };
    compute.output(TensorName::new("out"), sum);
    Arc::new(compute.finish().unwrap())
}
