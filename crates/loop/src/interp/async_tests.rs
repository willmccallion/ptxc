//! Asynchronous copies: each lands only at the wait that completes its
//! group, no thread may touch its elements before, and other threads see
//! them only after a barrier.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

use super::tests::{Y, float, input, shared, thread, with_body};
use super::*;
use crate::expr::ExprId;
use crate::stmt::{Condition, Stmt, ValueExpr};
use crate::test_kernels::kernel;
use ptxc_core::{Dim, Idx, TensorData};

/// The test kernel's input buffer.
const X: usize = 0;

/// The test kernel, a shared tile of `size` floats, and each thread's
/// first element of a pair, `2 * threadIdx.x`.
fn tiled(size: u64) -> (LoweredKernel, BufferId, ExprId) {
    let mut base = kernel();
    let tile = shared(&mut base, size);
    let lane = thread(&mut base);
    let two = base.exprs.constant(2);
    let pair = base.exprs.mul(lane, two);
    (base, tile, pair)
}

/// Each thread's copy of its pair of `x` into `tile` from `offset`, zero
/// where `bounds` fail.
fn copy_pair(
    base: &mut LoweredKernel,
    (tile, offset): (BufferId, i64),
    pair: ExprId,
    bounds: Vec<Condition>,
) -> Stmt {
    let offset = base.exprs.constant(offset);
    let target_index = base.exprs.add(pair, offset);
    let source = BufferId::new(X);
    Stmt::AsyncCopy {
        target: tile,
        target_index,
        source,
        source_index: pair,
        run: Dim::new(2),
        bounds,
    }
}

/// Each thread's store of `tile`'s elements `from + 2 * threadIdx.x` and
/// the next into its pair of `y`.
fn store_pair(base: &mut LoweredKernel, tile: BufferId, (pair, from): (ExprId, i64)) -> Stmt {
    let stores = (0..2)
        .map(|step| {
            let element = base.exprs.constant(step);
            let at = base.exprs.add(pair, element);
            let offset = base.exprs.constant(from);
            let read = base.exprs.add(at, offset);
            let value = ValueExpr::Load { buffer: tile, index: read };
            Stmt::Store { buffer: BufferId::new(Y), index: at, value }
        })
        .collect();
    Stmt::Seq(stores)
}

fn in_tile(tile: BufferId, stmts: Vec<Stmt>) -> Stmt {
    Stmt::Alloc { buffer: tile, body: Box::new(Stmt::Seq(stmts)) }
}

fn floats(outputs: &[HostTensor]) -> Vec<f32> {
    let TensorData::F32(values) = outputs[0].data() else { panic!() };
    values.clone()
}

#[test]
fn a_copy_lands_at_the_wait_for_its_group() {
    let (mut base, tile, pair) = tiled(8);
    let copy = copy_pair(&mut base, (tile, 0), pair, Vec::new());
    let read = store_pair(&mut base, tile, (pair, 0));
    let body = in_tile(tile, vec![copy, Stmt::AsyncCommit, Stmt::AsyncWait { pending: 0 }, read]);

    let outputs = run(&with_body(&base, body), &[input()]).unwrap();

    assert_eq!(floats(&outputs), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn reading_a_copied_element_before_a_wait_lands_it_is_an_error() {
    let (mut base, tile, pair) = tiled(8);
    let copy = copy_pair(&mut base, (tile, 0), pair, Vec::new());
    let read = store_pair(&mut base, tile, (pair, 0));
    let uncommitted = in_tile(tile, vec![copy.clone(), read.clone()]);
    let unwaited = in_tile(tile, vec![copy, Stmt::AsyncCommit, read]);

    let errors = [uncommitted, unwaited].map(|body| run(&with_body(&base, body), &[input()]).err());

    let in_flight = RunError::InFlight { buffer: TensorName::new("tile"), index: 0 };
    assert_eq!(errors, [Some(in_flight.clone()), Some(in_flight)]);
}

#[test]
fn a_wait_lands_every_group_but_the_newest_it_leaves_pending() {
    let (mut base, tile, pair) = tiled(16);
    let first = copy_pair(&mut base, (tile, 0), pair, Vec::new());
    let second = copy_pair(&mut base, (tile, 8), pair, Vec::new());
    let wait = Stmt::AsyncWait { pending: 1 };
    let groups = vec![first, Stmt::AsyncCommit, second, Stmt::AsyncCommit, wait];
    let program = |read: Stmt| {
        let mut stmts = groups.clone();
        stmts.extend([read, Stmt::AsyncWait { pending: 0 }]);
        in_tile(tile, stmts)
    };
    let older = program(store_pair(&mut base, tile, (pair, 0)));
    let newer = program(store_pair(&mut base, tile, (pair, 8)));

    let landed = run(&with_body(&base, older), &[input()]);
    let pending = run(&with_body(&base, newer), &[input()]);

    assert!(landed.is_ok(), "{:?}", landed.err());
    assert_eq!(
        pending.err(),
        Some(RunError::InFlight { buffer: TensorName::new("tile"), index: 8 })
    );
}

#[test]
fn a_commit_with_no_copies_is_a_group_a_wait_counts() {
    let (mut base, tile, pair) = tiled(8);
    let copy = copy_pair(&mut base, (tile, 0), pair, Vec::new());
    let read = store_pair(&mut base, tile, (pair, 0));
    let commits = |count| vec![Stmt::AsyncCommit; count];
    let program = |count| {
        let mut stmts = vec![copy.clone()];
        stmts.extend(commits(count));
        stmts.extend([Stmt::AsyncWait { pending: 1 }, read.clone()]);
        in_tile(tile, stmts)
    };

    let after_empty = run(&with_body(&base, program(2)), &[input()]);
    let alone = run(&with_body(&base, program(1)), &[input()]);

    assert!(after_empty.is_ok(), "{:?}", after_empty.err());
    assert_eq!(alone.err(), Some(RunError::InFlight { buffer: TensorName::new("tile"), index: 0 }));
}

#[test]
fn a_copy_whose_bounds_fail_writes_zeros() {
    let (mut base, tile, pair) = tiled(8);
    let lane = thread(&mut base);
    let half = base.exprs.constant(2);
    let first_half = Condition { index: lane, bound: half };
    let copy = copy_pair(&mut base, (tile, 0), pair, vec![first_half]);
    let read = store_pair(&mut base, tile, (pair, 0));
    let body = in_tile(tile, vec![copy, Stmt::AsyncCommit, Stmt::AsyncWait { pending: 0 }, read]);

    let outputs = run(&with_body(&base, body), &[input()]).unwrap();

    assert_eq!(floats(&outputs), [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn other_threads_read_a_landed_copy_only_after_a_barrier() {
    let (mut base, tile, pair) = tiled(8);
    let copy = copy_pair(&mut base, (tile, 0), pair, Vec::new());
    let zero = base.exprs.constant(0);
    let lane = thread(&mut base);
    let read_first = Stmt::Store {
        buffer: BufferId::new(Y),
        index: lane,
        value: ValueExpr::Load { buffer: tile, index: zero },
    };
    let program = |between: Vec<Stmt>| {
        let mut stmts = vec![copy.clone(), Stmt::AsyncCommit, Stmt::AsyncWait { pending: 0 }];
        stmts.extend(between);
        stmts.extend([read_first.clone(), base.body.clone()]);
        in_tile(tile, stmts)
    };

    let racing = run(&with_body(&base, program(Vec::new())), &[input()]);
    let ordered = run(&with_body(&base, program(vec![Stmt::Barrier])), &[input()]);

    assert_eq!(racing.err(), Some(RunError::Race { buffer: TensorName::new("tile"), index: 0 }));
    assert!(ordered.is_ok(), "{:?}", ordered.err());
}

#[test]
fn a_copy_into_an_element_another_thread_read_in_the_phase_races() {
    let (mut base, tile, pair) = tiled(8);
    let lane = thread(&mut base);
    let zero = base.exprs.constant(0);
    let set_own = Stmt::Store { buffer: tile, index: lane, value: float(1.0) };
    let read_first = Stmt::Store {
        buffer: BufferId::new(Y),
        index: lane,
        value: ValueExpr::Load { buffer: tile, index: zero },
    };
    let copy = copy_pair(&mut base, (tile, 0), pair, Vec::new());
    let stmts = vec![set_own, Stmt::Barrier, read_first, copy];

    let error = run(&with_body(&base, in_tile(tile, stmts)), &[input()]);

    assert_eq!(error.err(), Some(RunError::Race { buffer: TensorName::new("tile"), index: 0 }));
}

#[test]
fn freeing_a_tile_with_copies_in_flight_is_an_error() {
    let (mut base, tile, pair) = tiled(8);
    let copy = copy_pair(&mut base, (tile, 0), pair, Vec::new());
    let body = in_tile(tile, vec![copy, Stmt::AsyncCommit, base.body.clone()]);

    let error = run(&with_body(&base, body), &[input()]);

    assert_eq!(error.err(), Some(RunError::Unlanded(TensorName::new("tile"))));
}
