//! Errors only a hand-built kernel reaches: unset reads, unwritten outputs,
//! races, and indices that cannot be evaluated.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

use super::value::Value;
use super::*;
use crate::compute::Scalar;
use crate::expr::ExprId;
use crate::stmt::{Buffer, MemScope, Stmt, ValueExpr};
use crate::test_kernels::kernel;
use ptxc_core::{ElementCount, Idx, TensorData};

/// The output buffer of the test kernel.
pub(super) const Y: usize = 1;

pub(super) fn input() -> HostTensor {
    HostTensor::vector(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into())
}

pub(super) fn with_body(base: &LoweredKernel, body: Stmt) -> LoweredKernel {
    LoweredKernel { body, ..base.clone() }
}

/// `threadIdx.x`'s variable.
pub(super) fn thread(base: &mut LoweredKernel) -> ExprId {
    let (var, _) = base
        .vars
        .iter_enumerated()
        .find(|(_, var)| {
            matches!(
                var,
                crate::stmt::KernelVar::Thread { axis: crate::schedule::ThreadAxis::ThreadX, .. }
            )
        })
        .unwrap();
    base.exprs.var(var)
}

pub(super) fn shared(base: &mut LoweredKernel, size: u64) -> BufferId {
    base.buffers.push(Buffer {
        name: TensorName::new("tile"),
        dtype: DType::F32,
        scope: MemScope::Shared,
        size: ElementCount::new(size),
    })
}

pub(super) fn float(value: f64) -> ValueExpr {
    ValueExpr::Const(Scalar::float(DType::F32, value).unwrap())
}

#[test]
fn the_lowered_kernel_runs() {
    let outputs = run(&kernel(), &[input()]).unwrap();

    let TensorData::F32(values) = outputs[0].data() else { panic!() };
    assert_eq!(values, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn an_output_no_thread_stores_is_an_error() {
    let base = kernel();

    let error = run(&with_body(&base, Stmt::Seq(Vec::new())), &[input()]);

    assert_eq!(error.err(), Some(RunError::Unwritten { buffer: TensorName::new("y"), index: 0 }));
}

#[test]
fn reading_an_element_before_storing_it_is_an_error() {
    let mut base = kernel();
    let tile = shared(&mut base, 4);
    let lane = thread(&mut base);
    let read = Stmt::Store {
        buffer: BufferId::new(Y),
        index: lane,
        value: ValueExpr::Load { buffer: tile, index: lane },
    };
    let body = Stmt::Alloc { buffer: tile, body: Box::new(read) };

    let error = run(&with_body(&base, body), &[input()]);

    assert_eq!(error.err(), Some(RunError::Unset { buffer: TensorName::new("tile"), index: 0 }));
}

#[test]
fn threads_sharing_an_element_without_a_barrier_race_and_with_one_do_not() {
    let mut base = kernel();
    let tile = shared(&mut base, 4);
    let lane = thread(&mut base);
    let zero = base.exprs.constant(0);
    let write_own = Stmt::Store { buffer: tile, index: lane, value: float(1.0) };
    let read_first = Stmt::Store {
        buffer: BufferId::new(Y),
        index: lane,
        value: ValueExpr::Load { buffer: tile, index: zero },
    };
    let program = |between: Vec<Stmt>| {
        let mut stmts = vec![write_own.clone()];
        stmts.extend(between);
        stmts.push(read_first.clone());
        stmts.push(base.body.clone());
        Stmt::Alloc { buffer: tile, body: Box::new(Stmt::Seq(stmts)) }
    };

    let racing = run(&with_body(&base, program(Vec::new())), &[input()]);
    let ordered = run(&with_body(&base, program(vec![Stmt::Barrier])), &[input()]);

    assert_eq!(racing.err(), Some(RunError::Race { buffer: TensorName::new("tile"), index: 0 }));
    assert!(ordered.is_ok(), "{:?}", ordered.err());
}

#[test]
fn two_threads_writing_one_global_element_race() {
    let mut base = kernel();
    let zero = base.exprs.constant(0);
    let body = Stmt::Seq(vec![
        base.body.clone(),
        Stmt::Store { buffer: BufferId::new(Y), index: zero, value: float(0.0) },
    ]);

    let error = run(&with_body(&base, body), &[input()]);

    assert_eq!(error.err(), Some(RunError::Race { buffer: TensorName::new("y"), index: 0 }));
}

#[test]
fn an_index_that_divides_by_zero_is_an_error() {
    let mut base = kernel();
    let (one, zero) = (base.exprs.constant(1), base.exprs.constant(0));
    let undefined = base.exprs.floor_div(one, zero);
    let body = Stmt::Store { buffer: BufferId::new(Y), index: undefined, value: float(0.0) };

    assert_eq!(run(&with_body(&base, body), &[input()]).err(), Some(RunError::Index));
}

#[test]
fn a_thread_rereading_its_own_shared_element_does_not_race() {
    let mut base = kernel();
    let tile = shared(&mut base, 4);
    let lane = thread(&mut base);
    let own = ValueExpr::Load { buffer: tile, index: lane };
    let body = Stmt::Alloc {
        buffer: tile,
        body: Box::new(Stmt::Seq(vec![
            Stmt::Store { buffer: tile, index: lane, value: float(1.0) },
            Stmt::Store { buffer: tile, index: lane, value: own.clone() },
            Stmt::Store { buffer: BufferId::new(Y), index: lane, value: own },
            base.body.clone(),
        ])),
    };

    assert!(run(&with_body(&base, body), &[input()]).is_ok());
}

#[test]
fn a_read_by_one_thread_then_a_write_by_another_races() {
    let mut base = kernel();
    let tile = shared(&mut base, 4);
    let lane = thread(&mut base);
    let (zero, one) = (base.exprs.constant(0), base.exprs.constant(1));
    let only_lane_one = |stmt| Stmt::If {
        condition: crate::stmt::Condition { index: lane, bound: one },
        then: Box::new(stmt),
    };
    let body = Stmt::Alloc {
        buffer: tile,
        body: Box::new(Stmt::Seq(vec![
            Stmt::Store { buffer: tile, index: zero, value: float(1.0) },
            Stmt::Barrier,
            Stmt::Store {
                buffer: BufferId::new(Y),
                index: lane,
                value: ValueExpr::Load { buffer: tile, index: zero },
            },
            only_lane_one(Stmt::Store { buffer: tile, index: zero, value: float(2.0) }),
            base.body.clone(),
        ])),
    };

    let error = run(&with_body(&base, body), &[input()]);

    assert_eq!(error.err(), Some(RunError::Race { buffer: TensorName::new("tile"), index: 0 }));
}

#[test]
fn an_index_at_the_buffers_size_is_out_of_bounds() {
    let mut base = kernel();
    let past = base.exprs.constant(8);
    let body = Stmt::Store { buffer: BufferId::new(Y), index: past, value: float(0.0) };

    let error = run(&with_body(&base, body), &[input()]);

    assert_eq!(error.err(), Some(RunError::OutOfBounds { buffer: TensorName::new("y"), index: 8 }));
}

#[test]
fn threads_of_different_blocks_writing_one_global_element_race() {
    let mut base = crate::test_kernels::two_block_kernel();
    let lane = thread(&mut base);
    let body = Stmt::Store { buffer: BufferId::new(Y), index: lane, value: float(0.0) };

    let error = run(&with_body(&base, body), &[input()]);

    assert_eq!(error.err(), Some(RunError::Race { buffer: TensorName::new("y"), index: 0 }));
    assert!(run(&base, &[input()]).is_ok());

    let (block, _) = base
        .vars
        .iter_enumerated()
        .find(|(_, var)| {
            matches!(
                var,
                crate::stmt::KernelVar::Thread { axis: crate::schedule::ThreadAxis::BlockX, .. }
            )
        })
        .unwrap();
    let block = base.exprs.var(block);
    let shifted = base.exprs.add(lane, block);
    let neighbours = Stmt::Store { buffer: BufferId::new(Y), index: shifted, value: float(0.0) };
    let error = run(&with_body(&base, neighbours), &[input()]);
    assert_eq!(error.err(), Some(RunError::Race { buffer: TensorName::new("y"), index: 1 }));
}

/// `a * b + c` is `1 + 2^-23 + 2^-24 - 2^-70`, just below halfway between
/// two f32s. Rounded once it goes down; rounded first to f64 it would lose
/// the `2^-70`, land on the halfway point and tie to the even neighbour
/// above.
#[test]
fn a_multiply_add_rounds_once_where_rounding_through_f64_would_round_twice() {
    let (epsilon, scale) = (f64::from(f32::EPSILON), 2.0_f64.powi(-12));
    let factors = [scale * (1.0 + epsilon), scale * (1.0 - epsilon)].map(Value::Float);

    let fused = super::ops::mul_add(factors[0], factors[1], Value::Float(1.0 + epsilon));

    assert_eq!(fused, Value::Float(1.0 + epsilon));
}
