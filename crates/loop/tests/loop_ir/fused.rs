//! An f32 sum of products fused into multiply-adds, each rounded once.

use crate::compute_build::{binary, load, shape};
use crate::interp::{floats, run_default, tensor};
use crate::schedule_support::open;
use ptxc_core::{DType, Dim};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeDef, ComputeName, ReduceOp, TensorExpr, TensorName, VarName,
    print_kernel,
};
use std::sync::Arc;

/// `out[i] = sum over k of x[i, k] * y[i, k]` on `dtype`.
fn dot(dtype: DType, columns: u64) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("dot"));
    let x = compute.input(TensorName::new("x"), dtype, shape(&[1, columns]));
    let y = compute.input(TensorName::new("y"), dtype, shape(&[1, columns]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(1));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(columns));
    let product = binary(BinaryOp::Mul, load(x, &[i, k]), load(y, &[i, k]));
    compute.output(
        TensorName::new("out"),
        TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body: Box::new(product) },
    );
    Arc::new(compute.finish().unwrap())
}

#[test]
fn a_summed_f32_product_rounds_once() {
    let epsilon = f32::EPSILON;
    let x = tensor(&[1, 2], vec![1.0, 1.0 + epsilon]);
    let y = tensor(&[1, 2], vec![-1.0, 1.0 - epsilon]);

    let outputs = run_default(&dot(DType::F32, 2), &[x, y]).unwrap();

    assert_eq!(floats(&outputs[0]), [-(epsilon * epsilon)]);
}

#[test]
fn only_an_f32_sum_of_a_product_is_fused() {
    let lowered = |dtype| {
        let (schedule, _) = open(&dot(dtype, 4));
        print_kernel(&crate::lower::lower(schedule))
    };

    assert!(lowered(DType::F32).contains("= fma(%x["));
    assert!(lowered(DType::I32).contains("= add(%acc[0], mul(%x["));
}
