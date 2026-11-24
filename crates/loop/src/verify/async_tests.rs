//! The rules an asynchronous copy keeps, each broken by hand.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

use super::*;
use crate::compute::TensorName;
use crate::stmt::{Buffer, Condition, LoweredKernel};
use crate::test_kernels::kernel;
use ptxc_core::{Dim, ElementCount, Idx};

const X: usize = 0;
const Y: usize = 1;

/// A buffer of `size` elements of `dtype` in `scope`, named `tile`.
fn buffer(kernel: &mut LoweredKernel, scope: MemScope, dtype: DType, size: u64) -> BufferId {
    let (name, size) = (TensorName::new("tile"), ElementCount::new(size));
    kernel.buffers.push(Buffer { name, dtype, scope, size })
}

/// `threadIdx.x` times `factor`, plus `offset`.
fn per_thread(kernel: &mut LoweredKernel, factor: i64, offset: i64) -> ExprId {
    let (var, _) = kernel
        .vars
        .iter_enumerated()
        .find(|(_, var)| matches!(var, KernelVar::Thread { .. }))
        .unwrap();
    let lane = kernel.exprs.var(var);
    let (factor, offset) = (kernel.exprs.constant(factor), kernel.exprs.constant(offset));
    let scaled = kernel.exprs.mul(lane, factor);
    kernel.exprs.add(scaled, offset)
}

/// `kernel` running only an asynchronous copy of `run` elements from
/// `source` to `target`, with `target` allocated around it if it is not
/// global.
fn copying(
    kernel: &LoweredKernel,
    [(target, target_index), (source, source_index)]: [(BufferId, ExprId); 2],
    run: u64,
    bounds: Vec<Condition>,
) -> LoweredKernel {
    let run = Dim::new(run);
    let copy = Stmt::AsyncCopy { target, target_index, source, source_index, run, bounds };
    let body = if kernel.buffers[target].scope == MemScope::Global {
        copy
    } else {
        Stmt::Alloc { buffer: target, body: Box::new(copy) }
    };
    LoweredKernel { body, ..kernel.clone() }
}

#[test]
fn a_copy_goes_from_a_global_input_to_shared_memory() {
    let mut base = kernel();
    let tile = buffer(&mut base, MemScope::Shared, DType::F32, 8);
    let local = buffer(&mut base, MemScope::Local, DType::F32, 8);
    let pair = per_thread(&mut base, 2, 0);
    let input = (BufferId::new(X), pair);

    let into = |target: BufferId, source: (BufferId, ExprId)| {
        verify(&copying(&base, [(target, pair), source], 2, Vec::new()))
    };

    assert_eq!(into(tile, input), Ok(()));
    let tile_name = TensorName::new("tile");
    assert_eq!(into(local, input), Err(VerifyError::AsyncTarget(tile_name.clone())));
    assert_eq!(into(BufferId::new(Y), input), Err(VerifyError::AsyncTarget(TensorName::new("y"))));
    assert_eq!(into(tile, (tile, pair)), Err(VerifyError::AsyncSource(tile_name)));
    let from_output = into(tile, (BufferId::new(Y), pair));
    assert_eq!(from_output, Err(VerifyError::LoadFromOutput(TensorName::new("y"))));
}

#[test]
fn a_copy_is_4_8_or_16_bytes_of_its_targets_type() {
    let mut base = kernel();
    let tile = buffer(&mut base, MemScope::Shared, DType::F32, 64);
    let halves = buffer(&mut base, MemScope::Shared, DType::F16, 64);
    let start = per_thread(&mut base, 16, 0);
    let copy = |target, run| {
        verify(&copying(&base, [(target, start), (BufferId::new(X), start)], run, Vec::new()))
    };

    let sizes = [1, 2, 4].map(|run| copy(tile, run));
    let rejected = [3, 8].map(|run| copy(tile, run));

    assert_eq!(sizes, [Ok(()), Ok(()), Ok(())]);
    assert_eq!(rejected, [12, 32].map(|bytes| Err(VerifyError::AsyncRun { bytes })));
    let place = "an asynchronous copy to tile".to_owned();
    let mixed = VerifyError::WrongType { place, expected: DType::F16, actual: DType::F32 };
    assert_eq!(copy(halves, 2), Err(mixed));
}

#[test]
fn both_runs_of_a_copy_start_at_a_multiple_of_its_length() {
    let mut base = kernel();
    let tile = buffer(&mut base, MemScope::Shared, DType::F32, 32);
    let aligned = per_thread(&mut base, 4, 0);
    let shifted = per_thread(&mut base, 4, 2);
    let lane = per_thread(&mut base, 4, 0);
    let eight = base.exprs.constant(8);
    let wrapped = base.exprs.floor_mod(lane, eight);
    let copy = |target_index, source_index| {
        let ends = [(tile, target_index), (BufferId::new(X), source_index)];
        verify(&copying(&base, ends, 4, Vec::new()))
    };

    let unaligned = Err(VerifyError::AsyncUnaligned(TensorName::new("tile")));
    assert_eq!(copy(aligned, aligned), Ok(()));
    assert_eq!(copy(aligned, wrapped), Ok(()));
    assert_eq!(copy(shifted, aligned), unaligned);
    assert_eq!(copy(aligned, shifted), unaligned);
}

#[test]
fn a_copys_run_stays_inside_its_tile_and_reads_only_bound_variables() {
    let mut base = kernel();
    let fits = buffer(&mut base, MemScope::Shared, DType::F32, 8);
    let short = buffer(&mut base, MemScope::Shared, DType::F32, 7);
    let pair = per_thread(&mut base, 2, 0);
    let unbound = base.vars.push(KernelVar::Loop(crate::compute::VarName::new("free")));
    let free = base.exprs.var(unbound);
    let condition = Condition { index: free, bound: pair };
    let copy = |target, bounds: Vec<Condition>| {
        verify(&copying(&base, [(target, pair), (BufferId::new(X), pair)], 2, bounds))
    };

    assert_eq!(copy(fits, Vec::new()), Ok(()));
    assert_eq!(copy(short, Vec::new()), Err(VerifyError::OutOfBounds(TensorName::new("tile"))));
    let free_name = crate::compute::VarName::new("free");
    assert_eq!(copy(fits, vec![condition]), Err(VerifyError::UnboundVar(free_name)));
}
