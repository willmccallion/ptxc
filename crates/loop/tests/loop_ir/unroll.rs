//! Expanding unrolled loops: copies with their values substituted, guards
//! the ranges decide resolved, and the results unchanged.

use crate::compute_build::{load, shape};
use crate::interp::{ints, shaped, target, tensor};
use crate::schedule_support::open;
use proptest::prelude::*;
use ptxc_core::{DType, Dim};
use ptxc_loop::{
    ComputeBuilder, ComputeDef, ComputeName, Primitive, ReduceOp, SplitFactor, TensorExpr,
    TensorName, ThreadAxis, VarName, print_kernel, run, unrolled, verify,
};
use ptxc_testkit::golden;
use std::sync::Arc;

/// `out[i] = sum over k of x[i, k]` on `i32`.
fn row_sums(rows: u64, columns: u64) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("row_sums"));
    let x = compute.input(TensorName::new("x"), DType::I32, shape(&[rows, columns]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(rows));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(columns));
    let body = Box::new(load(x, &[i, k]));
    compute.output(
        TensorName::new("out"),
        TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body },
    );
    Arc::new(compute.finish().unwrap())
}

#[test]
fn an_unrolled_split_resolves_the_guards_its_values_decide() {
    let compute = row_sums(2, 4);
    let (mut schedule, nest) = open(&compute);
    let parts = schedule.split(nest[1], vec![SplitFactor::new(3).unwrap()]).unwrap();
    for loop_ in parts {
        schedule.apply(Primitive::Unroll { loop_ }).unwrap();
    }
    let kernel = schedule.validate(&target()).unwrap().lower();

    let expanded = unrolled(&kernel);

    verify(&expanded).unwrap();
    golden::check("lower/unrolled_row_sums.txt", &print_kernel(&expanded)).unwrap();
    let input = tensor(&[2, 4], (1..=8).collect::<Vec<i32>>());
    let sums = run(&expanded, std::slice::from_ref(&input)).unwrap();
    assert_eq!(ints(&sums[0]), [10, 26]);
}

#[test]
fn an_unrolled_copy_drops_a_guard_its_thread_range_decides() {
    let compute = row_sums(4, 2);
    let (mut schedule, nest) = open(&compute);
    let parts = schedule.split(nest[0], vec![SplitFactor::new(3).unwrap()]).unwrap();
    schedule.apply(Primitive::Bind { loop_: parts[1], axis: ThreadAxis::ThreadX }).unwrap();
    schedule.apply(Primitive::Unroll { loop_: parts[0] }).unwrap();
    let kernel = schedule.validate(&target()).unwrap().lower();

    let expanded = unrolled(&kernel);

    verify(&expanded).unwrap();
    golden::check("lower/unrolled_thread_guard.txt", &print_kernel(&expanded)).unwrap();
    let input = tensor(&[4, 2], (1..=8).collect::<Vec<i32>>());
    let sums = run(&expanded, std::slice::from_ref(&input)).unwrap();
    assert_eq!(ints(&sums[0]), [3, 7, 11, 15]);
}

proptest! {
    #![proptest_config(crate::cases(256))]

    /// Whatever loops a schedule unrolls, the expanded kernel verifies and
    /// computes what the rolled one does.
    #[test]
    fn unrolling_never_changes_a_result(
        extents in [1_u64..=4, 1_u64..=3, 1_u64..=6],
        steps in prop::collection::vec(crate::schedule_properties::step(), 0..6),
        binds in prop::collection::vec((0_usize..6, 0_usize..4), 0..3),
        unrolls in prop::collection::vec(0_usize..8, 1..=3),
        data in prop::collection::vec(-1000_i32..=1000, 72),
    ) {
        let compute = crate::interp::int_reduction(extents);
        let mut schedule = shaped(&compute, &steps, &binds);
        for position in unrolls {
            let nest = schedule.loops().nest();
            let loop_ = nest[position % nest.len()];
            let _rejected_or_applied = schedule.apply(Primitive::Unroll { loop_ });
        }
        if let Ok(validated) = schedule.validate(&target()) {
            let kernel = validated.lower();
            let count = usize::try_from(extents.iter().product::<u64>()).unwrap();
            let input = tensor(&extents, data[..count].to_vec());

            let expanded = unrolled(&kernel);

            prop_assert_eq!(verify(&expanded), Ok(()));
            let rolled = run(&kernel, std::slice::from_ref(&input)).unwrap();
            let expanded = run(&expanded, std::slice::from_ref(&input)).unwrap();
            prop_assert_eq!(ints(&expanded[0]), ints(&rolled[0]));
        }
    }
}
