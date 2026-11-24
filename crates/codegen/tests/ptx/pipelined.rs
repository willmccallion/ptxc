//! A pipelined loop divides for its slots once an iteration, before the
//! branches inside it, however many guarded loads read them.

use crate::ptxas::assemble;
use crate::target;
use ptxc_codegen::generate;
use ptxc_core::{DType, Dim, Shape};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeDef, ComputeName, ReduceOp, TensorExpr, TensorName, VarName,
};
use ptxc_testkit::pipelined::pipelined;
use std::sync::Arc;

/// `out[i, j] = sum over k of a[i, k] * b[k, j]` in f32 over `extents`.
fn product([rows, columns, depth]: [u64; 3]) -> Arc<ComputeDef> {
    let shape = |dims: &[u64]| Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap();
    let mut compute = ComputeBuilder::new(ComputeName::new("product"));
    let lhs = compute.input(TensorName::new("a"), DType::F32, shape(&[rows, depth]));
    let rhs = compute.input(TensorName::new("b"), DType::F32, shape(&[depth, columns]));
    let (_, row) = compute.spatial(VarName::new("i"), Dim::new(rows));
    let (_, column) = compute.spatial(VarName::new("j"), Dim::new(columns));
    let (axis, step) = compute.reduce_axis(VarName::new("k"), Dim::new(depth));
    let load = |tensor, index| Box::new(TensorExpr::Load { tensor, index });
    let (left, right) = (load(lhs, vec![row, step]), load(rhs, vec![step, column]));
    let body = Box::new(TensorExpr::Binary(BinaryOp::Mul, left, right));
    compute.output(
        TensorName::new("out"),
        TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![axis], body },
    );
    Arc::new(compute.finish().unwrap())
}

/// The PTX of `compute` under the default schedule, its staged loop
/// pipelined over `stages` instead of the default's.
fn pipelined_over(compute: &Arc<ComputeDef>, stages: u32) -> String {
    let kernel = pipelined(compute, stages, &target()).lower();
    let module = generate(&kernel, &target()).unwrap();
    assemble(&module);
    module.to_string()
}

#[test]
fn a_pipelined_loop_divides_for_its_slots_once_an_iteration() {
    // One row: every load of a thread's tile is under its own row guard.
    let module = pipelined_over(&product([1, 64, 768]), 3);

    // One remainder by 3 for the slot an iteration reads, which every
    // guarded load reuses, and one for the slot it refills.
    assert_eq!(module.matches("rem.u32").count(), 2, "{module}");
}
