//! Each rule the verifier enforces, broken by hand in a lowered kernel.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

use super::*;
use crate::compute::{BinaryOp, Scalar, TensorName};
use crate::stmt::{Condition, LoweredKernel};
use crate::test_kernels::kernel;
use ptxc_core::{Dim, Idx};

const X: usize = 0;
const Y: usize = 1;

fn with_body(kernel: &LoweredKernel, body: Stmt) -> LoweredKernel {
    LoweredKernel { body, ..kernel.clone() }
}

fn scratch(kernel: &mut LoweredKernel, scope: MemScope, dtype: DType, size: u64) -> BufferId {
    kernel.buffers.push(crate::stmt::Buffer {
        name: TensorName::new("scratch"),
        dtype,
        scope,
        size: ptxc_core::ElementCount::new(size),
    })
}

fn thread_var(kernel: &LoweredKernel) -> ExprId {
    let (var, _) = kernel
        .vars
        .iter_enumerated()
        .find(|(_, var)| matches!(var, KernelVar::Thread { .. }))
        .unwrap();
    kernel.exprs.clone().var(var)
}

fn float(value: f64) -> ValueExpr {
    ValueExpr::Const(Scalar::float(DType::F32, value).unwrap())
}

#[test]
fn a_lowered_kernel_verifies() {
    assert_eq!(verify(&kernel()), Ok(()));
}

#[test]
fn a_barrier_under_a_thread_dependent_condition_is_rejected() {
    let mut base = kernel();
    let thread = thread_var(&base);
    let one = base.exprs.constant(1);
    let body = Stmt::If {
        condition: Condition { index: thread, bound: one },
        then: Box::new(Stmt::Barrier),
    };

    assert_eq!(verify(&with_body(&base, body)), Err(VerifyError::DivergentBarrier));
    assert_eq!(verify(&with_body(&base, Stmt::Barrier)), Ok(()));
}

#[test]
fn inputs_are_only_read_and_outputs_only_written() {
    let mut base = kernel();
    let zero = base.exprs.constant(0);
    let store = Stmt::Store { buffer: BufferId::new(X), index: zero, value: float(0.0) };
    let load = Stmt::Store {
        buffer: BufferId::new(Y),
        index: zero,
        value: ValueExpr::Load { buffer: BufferId::new(Y), index: zero },
    };

    assert_eq!(
        verify(&with_body(&base, store)),
        Err(VerifyError::StoreToInput(TensorName::new("x")))
    );
    assert_eq!(
        verify(&with_body(&base, load)),
        Err(VerifyError::LoadFromOutput(TensorName::new("y")))
    );
}

#[test]
fn scratch_buffers_are_used_inside_their_allocation_and_in_bounds() {
    let mut base = kernel();
    let local = scratch(&mut base, MemScope::Local, DType::F32, 2);
    let (zero, two) = (base.exprs.constant(0), base.exprs.constant(2));
    let store = |index| Stmt::Store { buffer: local, index, value: float(1.0) };
    let alloc = |body| Stmt::Alloc { buffer: local, body: Box::new(body) };

    assert_eq!(
        verify(&with_body(&base, store(zero))),
        Err(VerifyError::NotAllocated(TensorName::new("scratch")))
    );
    assert_eq!(
        verify(&with_body(&base, alloc(store(two)))),
        Err(VerifyError::OutOfBounds(TensorName::new("scratch")))
    );
    assert_eq!(verify(&with_body(&base, alloc(store(zero)))), Ok(()));
    let global = Stmt::Alloc { buffer: BufferId::new(Y), body: Box::new(Stmt::Seq(Vec::new())) };
    assert_eq!(
        verify(&with_body(&base, global)),
        Err(VerifyError::GlobalAllocated(TensorName::new("y")))
    );
}

#[test]
fn a_condition_on_a_variable_tightens_its_range() {
    let mut base = kernel();
    let shared = scratch(&mut base, MemScope::Shared, DType::F32, 4);
    let thread = thread_var(&base);
    let (two, four) = (base.exprs.constant(2), base.exprs.constant(4));
    let partner = base.exprs.add(thread, two);
    let reversed = base.exprs.add(two, thread);
    let guarded = |index, bound| Stmt::If {
        condition: Condition { index, bound },
        then: Box::new(Stmt::Store { buffer: shared, index: partner, value: float(0.0) }),
    };
    let alloc = |body| with_body(&base, Stmt::Alloc { buffer: shared, body: Box::new(body) });

    assert_eq!(verify(&alloc(guarded(partner, four))), Ok(()));
    assert_eq!(verify(&alloc(guarded(reversed, four))), Ok(()));
    assert_eq!(
        verify(&alloc(guarded(thread, four))),
        Err(VerifyError::OutOfBounds(TensorName::new("scratch")))
    );
}

#[test]
fn a_variable_is_read_only_where_it_is_bound() {
    let mut base = kernel();
    let stray = base.vars.push(KernelVar::Loop(VarName::new("stray")));
    let index = base.exprs.var(stray);
    let store = Stmt::Store { buffer: BufferId::new(Y), index, value: float(0.0) };

    assert_eq!(
        verify(&with_body(&base, store)),
        Err(VerifyError::UnboundVar(VarName::new("stray")))
    );
}

#[test]
fn values_have_the_types_they_are_used_at() {
    let mut base = kernel();
    let zero = base.exprs.constant(0);
    let int = ValueExpr::Const(Scalar::int(DType::I64, 1).unwrap());
    let wrong_store = Stmt::Store { buffer: BufferId::new(Y), index: zero, value: int.clone() };
    let mixed = ValueExpr::Binary(BinaryOp::Add, Box::new(float(1.0)), Box::new(int.clone()));
    let mixed_store = Stmt::Store { buffer: BufferId::new(Y), index: zero, value: mixed };
    let bad_select = ValueExpr::Select {
        condition: Box::new(float(1.0)),
        then: Box::new(float(1.0)),
        otherwise: Box::new(float(0.0)),
    };
    let select_store = Stmt::Store { buffer: BufferId::new(Y), index: zero, value: bad_select };
    let branches = ValueExpr::Select {
        condition: Box::new(ValueExpr::Const(Scalar::bool(true))),
        then: Box::new(float(1.0)),
        otherwise: Box::new(int),
    };
    let branch_store = Stmt::Store { buffer: BufferId::new(Y), index: zero, value: branches };
    let float_index = Stmt::LoadIndex {
        var: base.vars.push(KernelVar::Loaded(VarName::new("g"))),
        buffer: BufferId::new(X),
        index: zero,
        body: Box::new(Stmt::Seq(Vec::new())),
    };

    assert!(matches!(
        verify(&with_body(&base, wrong_store)),
        Err(VerifyError::WrongType { expected: DType::F32, actual: DType::I64, .. })
    ));
    assert!(matches!(verify(&with_body(&base, mixed_store)), Err(VerifyError::Operation(_))));
    assert!(matches!(
        verify(&with_body(&base, select_store)),
        Err(VerifyError::WrongType { expected: DType::Bool, .. })
    ));
    assert!(matches!(
        verify(&with_body(&base, branch_store)),
        Err(VerifyError::WrongType { actual: DType::I64, .. })
    ));
    assert!(matches!(
        verify(&with_body(&base, float_index)),
        Err(VerifyError::WrongType { actual: DType::F32, .. })
    ));
}

#[test]
fn a_barrier_under_a_condition_on_a_loop_is_allowed() {
    let mut base = kernel();
    let looped = base.vars.push(KernelVar::Loop(VarName::new("l")));
    let (var, two) = (base.exprs.var(looped), base.exprs.constant(2));
    let guarded =
        Stmt::If { condition: Condition { index: var, bound: two }, then: Box::new(Stmt::Barrier) };
    let body = Stmt::For {
        var: looped,
        extent: Dim::new(4),
        kind: crate::stmt::ForKind::Serial,
        body: Box::new(guarded),
    };

    assert_eq!(verify(&with_body(&base, body)), Ok(()));
}

#[test]
fn a_condition_on_a_bare_variable_tightens_its_range() {
    let mut base = kernel();
    let shared = scratch(&mut base, MemScope::Shared, DType::F32, 2);
    let thread = thread_var(&base);
    let two = base.exprs.constant(2);
    let store = Stmt::Store { buffer: shared, index: thread, value: float(0.0) };
    let guarded = Stmt::If {
        condition: Condition { index: thread, bound: two },
        then: Box::new(store.clone()),
    };
    let alloc = |body| with_body(&base, Stmt::Alloc { buffer: shared, body: Box::new(body) });

    assert_eq!(verify(&alloc(guarded)), Ok(()));
    assert_eq!(verify(&alloc(store)), Err(VerifyError::OutOfBounds(TensorName::new("scratch"))));
}
