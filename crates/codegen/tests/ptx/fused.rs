//! An f32 sum of products selects one `fma.rn.f32` per term.

use crate::emitted;
use crate::ptxas::assemble;
use ptxc_core::{DType, Dim, Shape};
use ptxc_loop::{BinaryOp, ComputeBuilder, ComputeName, ReduceOp, TensorExpr, TensorName, VarName};
use std::sync::Arc;

#[test]
fn a_tiled_f32_matmul_multiplies_and_adds_in_one_fma() {
    let shape = Shape::new([64, 64].map(Dim::new).to_vec()).unwrap();
    let mut compute = ComputeBuilder::new(ComputeName::new("matmul"));
    let lhs = compute.input(TensorName::new("a"), DType::F32, shape.clone());
    let rhs = compute.input(TensorName::new("b"), DType::F32, shape);
    let (_, row) = compute.spatial(VarName::new("i"), Dim::new(64));
    let (_, column) = compute.spatial(VarName::new("j"), Dim::new(64));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(64));
    let load = |tensor, index| Box::new(TensorExpr::Load { tensor, index });
    let body =
        TensorExpr::Binary(BinaryOp::Mul, load(lhs, vec![row, k]), load(rhs, vec![k, column]));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body: Box::new(body) };
    compute.output(TensorName::new("out"), sum);

    let module = emitted(Arc::new(compute.finish().unwrap()));

    assemble(&module);
    let text = module.to_string();
    assert_eq!(text.matches("fma.rn.f32").count(), 8 * 4 * 4, "{text}");
    assert!(!text.contains("mul.rn.f32"), "{text}");
}
