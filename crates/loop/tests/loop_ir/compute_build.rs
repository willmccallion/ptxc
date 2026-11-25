//! Building compute definitions, and their text.

use ptxc_core::float::Overflow;
use ptxc_core::{DType, Dim, Shape};
use ptxc_loop::{
    ArgKind, BinaryOp, ComputeBuilder, ComputeDef, ComputeName, ExprId, InfSign, ReduceOp, Scalar,
    TensorExpr, TensorName, TensorParamId, TieBreak, UnaryOp, VarName, print_compute,
};
use ptxc_testkit::golden;

pub(crate) fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.iter().map(|&dim| Dim::new(dim)).collect()).unwrap()
}

pub(crate) fn load(tensor: TensorParamId, index: &[ExprId]) -> TensorExpr {
    TensorExpr::load(tensor, index.to_vec())
}

pub(crate) fn binary(op: BinaryOp, lhs: TensorExpr, rhs: TensorExpr) -> TensorExpr {
    TensorExpr::Binary(op, Box::new(lhs), Box::new(rhs))
}

pub(crate) fn f32_const(value: f64) -> TensorExpr {
    TensorExpr::Const(Scalar::float(DType::F32, value).unwrap())
}

fn builder(name: &str) -> ComputeBuilder {
    ComputeBuilder::new(ComputeName::new(name))
}

fn axis(name: &str) -> VarName {
    VarName::new(name)
}

fn tensor(name: &str) -> TensorName {
    TensorName::new(name)
}

fn add() -> ComputeDef {
    let mut compute = builder("add");
    let left = compute.input(tensor("x"), DType::F32, shape(&[4, 8]));
    let right = compute.input(tensor("y"), DType::F32, shape(&[4, 8]));
    let (_, i) = compute.spatial(axis("i"), Dim::new(4));
    let (_, j) = compute.spatial(axis("j"), Dim::new(8));
    compute.output(tensor("sum"), binary(BinaryOp::Add, load(left, &[i, j]), load(right, &[i, j])));
    compute.finish().unwrap()
}

fn matmul() -> ComputeDef {
    let mut compute = builder("matmul");
    let lhs = compute.input(tensor("a"), DType::F16, shape(&[64, 32]));
    let rhs = compute.input(tensor("b"), DType::F16, shape(&[32, 16]));
    let (_, i) = compute.spatial(axis("i"), Dim::new(64));
    let (_, j) = compute.spatial(axis("j"), Dim::new(16));
    let (k_axis, k) = compute.reduce_axis(axis("k"), Dim::new(32));
    let widen = |value| TensorExpr::Cast {
        to: DType::F32,
        overflow: Overflow::Saturate,
        value: Box::new(value),
    };
    let product = binary(BinaryOp::Mul, widen(load(lhs, &[i, k])), widen(load(rhs, &[k, j])));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body: Box::new(product) };
    compute.output(tensor("c"), sum);
    compute.finish().unwrap()
}

fn gather() -> ComputeDef {
    let mut compute = builder("gather");
    let data = compute.input(tensor("data"), DType::F32, shape(&[10, 4]));
    let indices = compute.input(tensor("indices"), DType::I64, shape(&[3]));
    let (_, n) = compute.spatial(axis("n"), Dim::new(3));
    let (_, c) = compute.spatial(axis("c"), Dim::new(4));
    let row = compute.loaded_index(axis("row"), indices, vec![n]);
    let ten = compute.exprs().constant(10);
    let wrapped = compute.exprs().floor_mod(row, ten);
    compute.output(tensor("out"), load(data, &[wrapped, c]));
    compute.finish().unwrap()
}

fn argmax() -> ComputeDef {
    let mut compute = builder("argmax");
    let x = compute.input(tensor("x"), DType::F32, shape(&[5, 7]));
    let (_, row) = compute.spatial(axis("row"), Dim::new(5));
    let (k_axis, k) = compute.reduce_axis(axis("k"), Dim::new(7));
    let argmax = TensorExpr::ArgReduce {
        kind: ArgKind::Max,
        tie: TieBreak::First,
        axis: k_axis,
        body: Box::new(load(x, &[row, k])),
    };
    compute.output(tensor("position"), argmax);
    compute.finish().unwrap()
}

/// Two outputs from one elementwise pass, which may share a definition.
fn sum_and_difference() -> ComputeDef {
    let mut compute = builder("sum_and_difference");
    let left = compute.input(tensor("x"), DType::I32, shape(&[6]));
    let right = compute.input(tensor("y"), DType::I32, shape(&[6]));
    let (_, i) = compute.spatial(axis("i"), Dim::new(6));
    compute.output(tensor("sum"), binary(BinaryOp::Add, load(left, &[i]), load(right, &[i])));
    compute
        .output(tensor("difference"), binary(BinaryOp::Sub, load(left, &[i]), load(right, &[i])));
    compute.finish().unwrap()
}

fn relu() -> ComputeDef {
    let mut compute = builder("relu");
    let x = compute.input(tensor("x"), DType::F32, shape(&[16]));
    let (_, i) = compute.spatial(axis("i"), Dim::new(16));
    let positive = binary(BinaryOp::Greater, load(x, &[i]), f32_const(0.0));
    let select = TensorExpr::Select {
        condition: Box::new(positive),
        then: Box::new(load(x, &[i])),
        otherwise: Box::new(f32_const(0.0)),
    };
    compute.output(tensor("y"), select);
    compute.finish().unwrap()
}

/// A strided window summed and exponentiated: `exp(sum[r](x[i * 2 + r]))`.
fn strided_sum() -> ComputeDef {
    let mut compute = builder("strided_sum");
    let x = compute.input(tensor("x"), DType::F32, shape(&[20]));
    let (_, i) = compute.spatial(axis("i"), Dim::new(8));
    let (r_axis, r) = compute.reduce_axis(axis("r"), Dim::new(3));
    let two = compute.exprs().constant(2);
    let doubled = compute.exprs().mul(i, two);
    let window = compute.exprs().add(doubled, r);
    let body = Box::new(load(x, &[window]));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![r_axis], body };
    compute.output(tensor("exp_sum"), TensorExpr::Unary(UnaryOp::Exp, Box::new(sum)));
    compute.finish().unwrap()
}

/// A test on a shifted, strided element: `isinf<negative>(x[(i + 1) * 2])`.
fn shifted_test() -> ComputeDef {
    let mut compute = builder("shifted_test");
    let x = compute.input(tensor("x"), DType::F32, shape(&[20]));
    let (_, i) = compute.spatial(axis("i"), Dim::new(8));
    let (one, two) = (compute.exprs().constant(1), compute.exprs().constant(2));
    let next = compute.exprs().add(i, one);
    let next_doubled = compute.exprs().mul(next, two);
    let negative = UnaryOp::IsInf(InfSign::Negative);
    let test = TensorExpr::Unary(negative, Box::new(load(x, &[next_doubled])));
    compute.output(tensor("negative_infinite"), test);
    compute.finish().unwrap()
}

#[test]
fn outputs_take_the_spatial_axes_as_their_shape_and_their_expression_type() {
    let matmul = matmul();
    let argmax = argmax();
    let pair = sum_and_difference();

    assert_eq!(matmul.output_shape(), &shape(&[64, 16]));
    assert_eq!(matmul.outputs()[0].dtype, DType::F32);
    assert_eq!(argmax.output_shape(), &shape(&[5]));
    assert_eq!(argmax.outputs()[0].dtype, DType::I64);
    let dtypes: Vec<DType> = pair.outputs().iter().map(|output| output.dtype).collect();
    assert_eq!(dtypes, [DType::I32, DType::I32]);
    assert_eq!(pair.output_shape(), &shape(&[6]));
}

#[test]
fn a_definition_prints_its_inputs_axes_and_outputs() {
    for (name, def) in [
        ("add", add()),
        ("matmul", matmul()),
        ("gather", gather()),
        ("argmax", argmax()),
        ("sum_and_difference", sum_and_difference()),
        ("relu", relu()),
        ("strided_sum", strided_sum()),
        ("shifted_test", shifted_test()),
    ] {
        golden::check(format!("compute/{name}.txt"), &print_compute(&def)).unwrap();
    }
}

#[test]
fn an_expression_finds_its_one_reduction_under_its_epilogue() {
    let sum = matmul().outputs()[0].expr.clone();
    let narrowed = TensorExpr::Cast {
        to: DType::F16,
        overflow: Overflow::Saturate,
        value: Box::new(sum.clone()),
    };

    assert_eq!(narrowed.reduction(), Some(&sum));
    assert!(matches!(argmax().outputs()[0].expr.reduction(), Some(TensorExpr::ArgReduce { .. })));
    assert_eq!(add().outputs()[0].expr.reduction(), None);
}
