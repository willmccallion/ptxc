//! The default schedules: how each kind of stage is spread over blocks and
//! threads, and the edge cases that stay serial.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod matmul;

/// Property tests run `default` cases, or `PROPTEST_CASES` when set, so a
/// longer run needs no code change (`ProptestConfig::with_cases` alone
/// ignores the variable).
fn cases(default: u32) -> proptest::prelude::ProptestConfig {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|cases| cases.parse().ok())
        .unwrap_or(default);
    proptest::prelude::ProptestConfig::with_cases(cases)
}

use ptxc_core::{DType, Dim, Shape};
use ptxc_loop::{
    ArgKind, ComputeBuilder, ComputeDef, ComputeName, IterVarId, LaunchShape, ReduceOp, TensorExpr,
    TensorName, TieBreak, VarName, print_kernel,
};
use ptxc_schedule::default_schedule;
use ptxc_target::{SmVersion, preset};
use ptxc_testkit::golden;
use std::sync::Arc;

fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap()
}

/// `y = x` over `dims`, or one of `x`'s reductions over its last
/// `reduced` axes.
fn stage(dims: &[u64], reduced: usize, reduction: Option<Reduction>) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("stage"));
    let x = compute.input(TensorName::new("x"), DType::F32, shape(dims));
    let kept = dims.len() - reduced;
    let mut index = Vec::new();
    let mut axes = Vec::new();
    for (position, &extent) in dims.iter().enumerate() {
        let name = VarName::new(format!("a{position}"));
        if position < kept {
            index.push(compute.spatial(name, Dim::new(extent)).1);
        } else {
            let (axis, var) = compute.reduce_axis(name, Dim::new(extent));
            index.push(var);
            axes.push(axis);
        }
    }
    let read = TensorExpr::Load { tensor: x, index };
    let expr = match reduction {
        Some(reduce) => reduce(&axes, read),
        None => read,
    };
    compute.output(TensorName::new("y"), expr);
    Arc::new(compute.finish().unwrap())
}

/// A reduction of a body over the given axes.
type Reduction = fn(&[IterVarId], TensorExpr) -> TensorExpr;

fn sum(axes: &[IterVarId], body: TensorExpr) -> TensorExpr {
    TensorExpr::Reduce { op: ReduceOp::Sum, axes: axes.to_vec(), body: Box::new(body) }
}

fn argmax(axes: &[IterVarId], body: TensorExpr) -> TensorExpr {
    TensorExpr::ArgReduce {
        kind: ArgKind::Max,
        tie: TieBreak::First,
        axis: axes[0],
        body: Box::new(body),
    }
}

/// The printed default schedule of `compute` and its launch.
fn scheduled(compute: Arc<ComputeDef>) -> (String, LaunchShape) {
    let target = preset(SmVersion::SM_89).unwrap();
    let schedule = default_schedule(compute, &target).unwrap();
    let kernel = schedule.lower();
    (print_kernel(&kernel), kernel.launch())
}

fn launch_dims(launch: LaunchShape) -> ([u32; 3], [u32; 3]) {
    (
        launch.grid.map(ptxc_target::BlockCount::get),
        launch.block.map(ptxc_target::ThreadsPerBlock::get),
    )
}

#[test]
fn an_elementwise_stage_fills_blocks_of_256_threads() {
    let (text, launch) = scheduled(stage(&[4, 300], 0, None));

    assert_eq!(launch_dims(launch), ([5, 1, 1], [256, 1, 1]));
    golden::check("elementwise.txt", &text).unwrap();
}

#[test]
fn a_small_output_uses_one_block_of_exactly_its_size() {
    let (_, launch) = scheduled(stage(&[3, 5], 0, None));

    assert_eq!(launch_dims(launch), ([1, 1, 1], [15, 1, 1]));
}

#[test]
fn a_reduction_gives_each_row_a_block_whose_threads_share_a_tree() {
    let (text, launch) = scheduled(stage(&[6, 2, 300], 2, Some(sum)));

    assert_eq!(launch_dims(launch), ([6, 1, 1], [256, 1, 1]));
    golden::check("reduction.txt", &text).unwrap();
}

#[test]
fn an_arg_reduction_shares_its_tree_too() {
    let (text, launch) = scheduled(stage(&[3, 40], 1, Some(argmax)));

    assert_eq!(launch_dims(launch), ([3, 1, 1], [40, 1, 1]));
    golden::check("arg_reduction.txt", &text).unwrap();
}

#[test]
fn empty_and_scalar_stages_stay_serial() {
    for compute in [stage(&[0, 4], 0, None), stage(&[], 0, None), stage(&[0, 5], 1, Some(sum))] {
        let (_, launch) = scheduled(compute);

        assert_eq!(launch_dims(launch), ([1, 1, 1], [1, 1, 1]));
    }
}

#[test]
fn an_empty_reduction_still_spreads_its_rows_over_blocks() {
    let (_, launch) = scheduled(stage(&[7, 0], 1, Some(sum)));

    assert_eq!(launch_dims(launch), ([7, 1, 1], [1, 1, 1]));
}
