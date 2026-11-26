//! Groups of several nodes lowered into fewer stages by inlining (D80):
//! each fused stage on the GPU matches the loop interpreter exactly on
//! inputs whose sums are exact, on tensor cores too, and the group's
//! outputs match the reference interpreter.

use crate::device::match_interpreter_exactly;
use ptxc_core::{Axis, DType, Dim, Evaluate, HostTensor, Shape};
use ptxc_graph::{
    Activation, BinaryOp, ConvAttrs, Dilation, Graph, GraphBuilder, Groups, Op, Padding,
    Partitioned, Stride, TransposeAttrs, UnaryOp, ValueId, WindowAxis,
};
use ptxc_interp::Interpreter;
use ptxc_testkit::compare::{Tolerance, compare};
use ptxc_testkit::lowered::run_partitioned;
use ptxc_testkit::tensorized::exact_tensor;

fn exact(dtype: DType, dims: &[u64], salt: u64) -> HostTensor {
    exact_tensor(dtype, Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap(), salt)
}

fn one(builder: &mut GraphBuilder, op: Op, operands: &[ValueId]) -> ValueId {
    builder.add_node(op, operands.to_vec()).unwrap()[0]
}

fn input(builder: &mut GraphBuilder, tensor: &HostTensor) -> ValueId {
    let ty = ptxc_graph::TensorType::new(
        tensor.dtype(),
        tensor.shape().clone(),
        ptxc_graph::Layout::RowMajor,
    );
    builder.add_input(ty)
}

/// Runs `graph`, every node in one group, stage by stage on the GPU, each
/// stage checked against the loop interpreter; the outputs checked against
/// the reference interpreter's; the number of stages the group took.
fn fused_on_gpu(graph: Graph, inputs: &[HostTensor]) -> usize {
    let context = ptxc_graph::text::print(&graph);
    let expected = Interpreter::new(&graph).evaluate(inputs).unwrap();
    let nodes = graph.topological_order().unwrap();
    let partitioned = Partitioned::from_groups(graph, &[nodes]).unwrap();
    let mut stages = 0;

    let actual = run_partitioned(&partitioned, inputs, |kernel, arguments| {
        stages += 1;
        match_interpreter_exactly(kernel, arguments, &context)
    })
    .unwrap();

    for (actual, expected) in actual.iter().zip(&expected) {
        compare(actual, expected, Tolerance::Exact)
            .unwrap_or_else(|mismatch| panic!("{mismatch}\n{context}"));
    }
    stages
}

#[test]
fn an_f16_product_runs_its_bias_and_relu_on_tensor_cores_as_one_kernel() {
    let inputs = [
        exact(DType::F16, &[64, 128], 1),
        exact(DType::F16, &[128, 96], 2),
        exact(DType::F16, &[96], 3),
    ];
    let mut builder = GraphBuilder::new();
    let [lhs, rhs, bias] = [0, 1, 2].map(|at| input(&mut builder, &inputs[at]));
    let product = one(&mut builder, Op::MatMul, &[lhs, rhs]);
    let biased = one(&mut builder, Op::Binary(BinaryOp::Add), &[product, bias]);
    let out = one(&mut builder, Op::Activation(Activation::Relu), &[biased]);

    assert_eq!(fused_on_gpu(builder.finish(vec![out]).unwrap(), &inputs), 1);
}

#[test]
fn an_epilogue_reading_its_product_twice_runs_as_one_kernel() {
    let operands = [exact(DType::F32, &[40, 72], 4), exact(DType::F32, &[72, 48], 5)];
    let mut builder = GraphBuilder::new();
    let [lhs, rhs] = [0, 1].map(|at| input(&mut builder, &operands[at]));
    let product = one(&mut builder, Op::MatMul, &[lhs, rhs]);
    let erf = one(&mut builder, Op::Unary(UnaryOp::Erf), &[product]);
    let out = one(&mut builder, Op::Binary(BinaryOp::Mul), &[product, erf]);

    assert_eq!(fused_on_gpu(builder.finish(vec![out]).unwrap(), &operands), 1);
}

#[test]
fn an_f16_product_reads_its_transposed_weight_in_place() {
    let operands = [exact(DType::F16, &[64, 128], 6), exact(DType::F16, &[96, 128], 7)];
    let mut builder = GraphBuilder::new();
    let [lhs, weight] = [0, 1].map(|at| input(&mut builder, &operands[at]));
    let permutation = vec![Axis::new(1), Axis::new(0)];
    let transposed = one(&mut builder, Op::Transpose(TransposeAttrs { permutation }), &[weight]);
    let out = one(&mut builder, Op::MatMul, &[lhs, transposed]);

    assert_eq!(fused_on_gpu(builder.finish(vec![out]).unwrap(), &operands), 1);
}

#[test]
fn an_f16_convolution_runs_its_residual_add_and_relu_on_tensor_cores_as_one_kernel() {
    let inputs = [
        exact(DType::F16, &[2, 16, 10, 10], 8),
        exact(DType::F16, &[32, 16, 3, 3], 9),
        exact(DType::F16, &[2, 32, 10, 10], 10),
    ];
    let mut builder = GraphBuilder::new();
    let [x, weights, residual] = [0, 1, 2].map(|at| input(&mut builder, &inputs[at]));
    let axis = WindowAxis {
        stride: Stride::new(1).unwrap(),
        dilation: Dilation::new(1).unwrap(),
        pad_begin: Padding::new(1),
        pad_end: Padding::new(1),
    };
    let attrs = ConvAttrs { axes: vec![axis, axis], groups: Groups::new(1).unwrap() };
    let conv = one(&mut builder, Op::Conv(attrs), &[x, weights]);
    let summed = one(&mut builder, Op::Binary(BinaryOp::Add), &[conv, residual]);
    let out = one(&mut builder, Op::Activation(Activation::Relu), &[summed]);

    assert_eq!(fused_on_gpu(builder.finish(vec![out]).unwrap(), &inputs), 1);
}
