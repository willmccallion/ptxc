//! Where `cache_read` refuses a stage: at apply, for what no later
//! primitive can change, and at validation, for where the stage ends up.

use crate::cache::{a, b, matmul, tiled};
use crate::compute_build::{binary, load, shape};
use crate::interp::target;
use crate::schedule_support::open;
use ptxc_core::{DType, Dim, Idx};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeDef, ComputeName, LoopId, Open, Primitive, ReduceOp, Schedule,
    ScheduleError, SplitFactor, TensorExpr, TensorName, TensorParamId, ThreadAxis, VarName,
};
use std::sync::Arc;

fn stage(
    schedule: &mut Schedule<Open>,
    tensor: TensorParamId,
    at: LoopId,
) -> Result<(), ScheduleError> {
    schedule.apply(Primitive::CacheRead { tensor, at })
}

fn validation_error(schedule: Schedule<Open>) -> ScheduleError {
    schedule.validate(&target()).unwrap_err()
}

fn split(schedule: &mut Schedule<Open>, loop_: LoopId, factor: u64) -> Vec<LoopId> {
    schedule.split(loop_, vec![SplitFactor::new(factor).unwrap()]).unwrap()
}

fn name(text: &str) -> TensorName {
    TensorName::new(text)
}

#[test]
fn an_input_the_definition_lacks_cannot_be_staged() {
    let compute = matmul([2, 2, 2]);
    let (mut schedule, nest) = open(&compute);

    let error = stage(&mut schedule, TensorParamId::new(2), nest[0]);

    assert_eq!(error, Err(ScheduleError::UnknownTensor(TensorParamId::new(2))));
}

#[test]
fn a_bound_loop_cannot_hold_a_stage() {
    let compute = matmul([8, 8, 8]);
    let (mut schedule, _) = tiled(&compute, [4, 4]);
    let block_loop = schedule.loops().nest()[0];

    let error = stage(&mut schedule, a(), block_loop);

    let at = VarName::new("i.0");
    assert_eq!(error, Err(ScheduleError::CacheNotSerial { tensor: name("a"), at }));
}

#[test]
fn an_input_is_staged_at_most_once() {
    let compute = matmul([8, 8, 8]);
    let (mut schedule, [ko, ki]) = tiled(&compute, [4, 4]);
    stage(&mut schedule, a(), ko).unwrap();

    let error = stage(&mut schedule, a(), ki);

    assert_eq!(error, Err(ScheduleError::AlreadyCached(name("a"))));
}

#[test]
fn an_input_no_load_reads_cannot_be_staged() {
    let mut compute = ComputeBuilder::new(ComputeName::new("gather"));
    let data = compute.input(name("data"), DType::F32, shape(&[4]));
    let indices = compute.input(name("indices"), DType::I64, shape(&[2]));
    let unread = compute.input(name("unread"), DType::F32, shape(&[2]));
    let (_, n) = compute.spatial(VarName::new("n"), Dim::new(2));
    let row = compute.loaded_index(VarName::new("row"), indices, vec![n]);
    compute.output(name("out"), load(data, &[row]));
    let compute = Arc::new(compute.finish().unwrap());
    let (mut schedule, nest) = open(&compute);

    assert_eq!(
        stage(&mut schedule, unread, nest[0]),
        Err(ScheduleError::CacheUnread(name("unread")))
    );
    assert_eq!(
        stage(&mut schedule, indices, nest[0]),
        Err(ScheduleError::CacheUnread(name("indices")))
    );
    assert_eq!(
        stage(&mut schedule, data, nest[0]),
        Err(ScheduleError::CacheGathered(name("data")))
    );
}

#[test]
fn an_input_read_at_two_indices_cannot_be_staged() {
    let mut compute = ComputeBuilder::new(ComputeName::new("symmetric"));
    let x = compute.input(name("x"), DType::F32, shape(&[3, 3]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(3));
    let (_, j) = compute.spatial(VarName::new("j"), Dim::new(3));
    compute.output(name("out"), binary(BinaryOp::Add, load(x, &[i, j]), load(x, &[j, i])));
    let compute = Arc::new(compute.finish().unwrap());
    let (mut schedule, nest) = open(&compute);

    let error = stage(&mut schedule, x, nest[0]);

    assert_eq!(error, Err(ScheduleError::CacheLoads(name("x"))));
}

#[test]
fn a_stage_whose_loop_is_later_split_fails_validation() {
    let compute = matmul([8, 8, 8]);
    let (mut schedule, [ko, _]) = tiled(&compute, [4, 4]);
    stage(&mut schedule, a(), ko).unwrap();
    let _parts = split(&mut schedule, ko, 2);

    let error = validation_error(schedule);

    let at = VarName::new("k.0");
    assert_eq!(error, ScheduleError::CacheNotSerial { tensor: name("a"), at });
}

/// `out[i] = sum over k of x[i] * y[i, k], plus x[i]`: `x` is read in the
/// reduction's body and after it.
fn scaled_sum_plus_input() -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("scaled_sum"));
    let x = compute.input(name("x"), DType::I32, shape(&[3]));
    let y = compute.input(name("y"), DType::I32, shape(&[3, 4]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(3));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(4));
    let body = Box::new(binary(BinaryOp::Mul, load(x, &[i]), load(y, &[i, k])));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body };
    compute.output(name("out"), binary(BinaryOp::Add, sum, load(x, &[i])));
    Arc::new(compute.finish().unwrap())
}

#[test]
fn an_input_also_read_after_the_reduction_is_staged_only_outside_its_loops() {
    let compute = scaled_sum_plus_input();
    let x = TensorParamId::new(0);
    let (mut at_reduction, nest) = open(&compute);
    stage(&mut at_reduction, x, nest[1]).unwrap();
    let (mut inside_reduction, nest) = open(&compute);
    inside_reduction.apply(Primitive::Reorder { order: vec![nest[1], nest[0]] }).unwrap();
    stage(&mut inside_reduction, x, nest[0]).unwrap();
    let (mut outside, nest) = open(&compute);
    stage(&mut outside, x, nest[0]).unwrap();

    let expected = ScheduleError::CacheOutsideReduction(name("x"));
    assert_eq!(validation_error(at_reduction), expected);
    assert_eq!(validation_error(inside_reduction), expected);
    assert!(outside.validate(&target()).is_ok());
}

#[test]
fn an_index_that_is_not_affine_in_the_tile_loops_fails_validation() {
    let compute = matmul([4, 4, 4]);
    let (mut schedule, nest) = open(&compute);
    let fused = schedule.fuse(vec![nest[0], nest[1]]).unwrap();
    let parts = split(&mut schedule, fused, 3);
    stage(&mut schedule, a(), parts[0]).unwrap();

    let error = validation_error(schedule);

    assert_eq!(error, ScheduleError::CacheNotAffine(name("a")));
}

#[test]
fn an_index_that_falls_as_a_tile_loop_rises_fails_validation() {
    let mut compute = ComputeBuilder::new(ComputeName::new("reverse"));
    let x = compute.input(name("x"), DType::F32, shape(&[4]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(4));
    let exprs = compute.exprs();
    let (minus_one, last) = (exprs.constant(-1), exprs.constant(3));
    let negated = exprs.mul(i, minus_one);
    let reversed = exprs.add(negated, last);
    compute.output(name("out"), load(x, &[reversed]));
    let compute = Arc::new(compute.finish().unwrap());
    let (mut schedule, nest) = open(&compute);
    let parts = split(&mut schedule, nest[0], 2);
    stage(&mut schedule, x, parts[0]).unwrap();

    let error = validation_error(schedule);

    assert_eq!(error, ScheduleError::CacheNotAffine(name("x")));
}

#[test]
fn a_stage_in_a_reduction_shared_between_threads_fails_validation() {
    let compute = matmul([8, 8, 8]);
    let (mut schedule, [ko, ki]) = tiled(&compute, [4, 4]);
    stage(&mut schedule, b(), ko).unwrap();
    schedule.apply(Primitive::Bind { loop_: ki, axis: ThreadAxis::ThreadZ }).unwrap();

    let error = validation_error(schedule);

    assert_eq!(error, ScheduleError::CacheCrossThread(name("b")));
}

/// A matmul whose `a` is staged whole: a one-row tile of `depth` `i32`s.
fn one_row_staged(depth: u64) -> Schedule<Open> {
    let compute = matmul([1, 1, depth]);
    let (mut schedule, nest) = open(&compute);
    stage(&mut schedule, a(), nest[0]).unwrap();
    schedule
}

#[test]
fn tiles_past_a_blocks_shared_memory_fail_validation() {
    let limit = target().limits().block.smem;
    let fitting = u64::from(limit.get()) / 4;

    let error = validation_error(one_row_staged(fitting + 1));

    let bytes = ptxc_core::ByteSize::new(u64::from(limit.get()) + 4);
    assert_eq!(error, ScheduleError::SharedTooLarge { bytes, limit });
    assert!(one_row_staged(fitting).validate(&target()).is_ok());
}
