//! Groups of several nodes lowered by inlining their stages into each other
//! (D80): how many stages each takes, and that run stage by stage in the
//! loop interpreter they agree with the reference interpreter.

use ptxc_core::float::Overflow;
use ptxc_core::{
    Affine, Axis, DType, Dim, ElementCount, Evaluate, Granularity, HostTensor, QuantDType,
    QuantParams, Scale, Shape, ZeroPoint,
};
use ptxc_graph::{
    Activation, BinaryOp, ConvAttrs, DequantizeAttrs, Dilation, Graph, GraphBuilder, Groups,
    KeepDims, Layout, NodeId, Op, Padding, Partitioned, QuantizeAttrs, ReduceAttrs, ReduceOp,
    SoftmaxAttrs, SoftmaxKind, Stride, TensorType, TransposeAttrs, UnaryOp, ValueId, WindowAxis,
};
use ptxc_interp::Interpreter;
use ptxc_testkit::compare::{Tolerance, compare};
use ptxc_testkit::lowered::run_partitioned;

fn f32s(dims: &[u64]) -> TensorType {
    TensorType::new(
        DType::F32,
        Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap(),
        Layout::RowMajor,
    )
}

/// Values in [-1, 1] no two neighbours share, offset by `seed`.
fn values(dims: &[u64], seed: u16) -> HostTensor {
    let shape = Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap();
    let count = u16::try_from(shape.element_count().get()).unwrap();
    let data: Vec<f32> = (0..count).map(|n| f32::from((n * 7 + seed) % 17) / 8.0 - 1.0).collect();
    HostTensor::new(shape, data.into()).unwrap()
}

fn one(builder: &mut GraphBuilder, op: Op, operands: &[ValueId]) -> ValueId {
    builder.add_node(op, operands.to_vec()).unwrap()[0]
}

/// `graph` with every node in one group: the number of stages that group
/// lowers to, after checking its outputs for `inputs` against the
/// reference interpreter's, within f32's tolerance for sums of `depth`.
fn fused_stages(graph: Graph, inputs: &[HostTensor], depth: u64) -> usize {
    let expected = Interpreter::new(&graph).evaluate(inputs).unwrap();
    let nodes: Vec<NodeId> = graph.topological_order().unwrap();
    let partitioned = Partitioned::from_groups(graph, &[nodes]).unwrap();
    let stages = ptxc_lower::lower(&partitioned).unwrap()[ptxc_core::Idx::new(0)].stages().len();
    let actual = run_partitioned(&partitioned, inputs, ptxc_loop::run).unwrap();
    let tolerance = Tolerance::for_dtype(DType::F32).for_reduction(ElementCount::new(depth));
    for (actual, expected) in actual.iter().zip(&expected) {
        compare(actual, expected, tolerance).unwrap();
    }
    stages
}

/// A product of `[4, 8]` by `[8, 6]`, its inputs, and the builder to add
/// its epilogue to.
fn product() -> (GraphBuilder, ValueId, Vec<HostTensor>) {
    let mut builder = GraphBuilder::new();
    let lhs = builder.add_input(f32s(&[4, 8]));
    let rhs = builder.add_input(f32s(&[8, 6]));
    let product = one(&mut builder, Op::MatMul, &[lhs, rhs]);
    (builder, product, vec![values(&[4, 8], 1), values(&[8, 6], 2)])
}

#[test]
fn a_product_runs_its_bias_and_relu_as_its_epilogue() {
    let (mut builder, product, mut inputs) = product();
    let bias = builder.add_input(f32s(&[6]));
    let biased = one(&mut builder, Op::Binary(BinaryOp::Add), &[product, bias]);
    let out = one(&mut builder, Op::Activation(Activation::Relu), &[biased]);
    inputs.push(values(&[6], 3));

    let stages = fused_stages(builder.finish(vec![out]).unwrap(), &inputs, 8);

    assert_eq!(stages, 1);
}

#[test]
fn an_epilogue_reading_its_product_twice_stays_one_stage() {
    let (mut builder, product, inputs) = product();
    let erf = one(&mut builder, Op::Unary(UnaryOp::Erf), &[product]);
    let out = one(&mut builder, Op::Binary(BinaryOp::Mul), &[product, erf]);

    let stages = fused_stages(builder.finish(vec![out]).unwrap(), &inputs, 8);

    assert_eq!(stages, 1);
}

#[test]
fn a_transposed_weight_is_read_in_place_by_its_product() {
    let mut builder = GraphBuilder::new();
    let lhs = builder.add_input(f32s(&[4, 8]));
    let weight = builder.add_input(f32s(&[6, 8]));
    let permutation = vec![Axis::new(1), Axis::new(0)];
    let transposed = one(&mut builder, Op::Transpose(TransposeAttrs { permutation }), &[weight]);
    let out = one(&mut builder, Op::MatMul, &[lhs, transposed]);
    let inputs = [values(&[4, 8], 4), values(&[6, 8], 5)];

    let stages = fused_stages(builder.finish(vec![out]).unwrap(), &inputs, 8);

    assert_eq!(stages, 1);
}

#[test]
fn an_elementwise_chain_and_the_reduction_it_feeds_run_as_one_stage() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32s(&[3, 10]));
    let negated = one(&mut builder, Op::Unary(UnaryOp::Neg), &[x]);
    let exp = one(&mut builder, Op::Unary(UnaryOp::Exp), &[negated]);
    let attrs =
        ReduceAttrs { op: ReduceOp::Sum, axes: vec![Axis::new(1)], keep_dims: KeepDims::Drop };
    let sum = one(&mut builder, Op::Reduce(attrs), &[exp]);
    let out = one(&mut builder, Op::Unary(UnaryOp::Log), &[sum]);

    let stages = fused_stages(builder.finish(vec![out]).unwrap(), &[values(&[3, 10], 6)], 10);

    assert_eq!(stages, 1);
}

#[test]
fn a_convolution_runs_its_residual_add_and_relu_as_its_epilogue() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32s(&[1, 2, 5, 5]));
    let weights = builder.add_input(f32s(&[3, 2, 3, 3]));
    let residual = builder.add_input(f32s(&[1, 3, 5, 5]));
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
    let inputs = [values(&[1, 2, 5, 5], 7), values(&[3, 2, 3, 3], 8), values(&[1, 3, 5, 5], 9)];

    let stages = fused_stages(builder.finish(vec![out]).unwrap(), &inputs, 18);

    assert_eq!(stages, 1);
}

#[test]
fn a_value_the_group_exports_is_still_written() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32s(&[6]));
    let negated = one(&mut builder, Op::Unary(UnaryOp::Neg), &[x]);
    let out = one(&mut builder, Op::Unary(UnaryOp::Exp), &[negated]);

    let stages = fused_stages(builder.finish(vec![out, negated]).unwrap(), &[values(&[6], 10)], 1);

    assert_eq!(stages, 2);
}

#[test]
fn a_product_its_consumer_reads_transposed_keeps_its_stage() {
    let (mut builder, product, inputs) = product();
    let permutation = vec![Axis::new(1), Axis::new(0)];
    let out = one(&mut builder, Op::Transpose(TransposeAttrs { permutation }), &[product]);

    let stages = fused_stages(builder.finish(vec![out]).unwrap(), &inputs, 8);

    assert_eq!(stages, 2);
}

#[test]
fn the_intermediates_of_each_node_keep_their_own_buffers() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32s(&[3, 7]));
    let softmax = SoftmaxAttrs { kind: SoftmaxKind::Softmax, axis: Axis::new(1) };
    let first = one(&mut builder, Op::Softmax(softmax), &[x]);
    let out = one(&mut builder, Op::Softmax(softmax), &[first]);

    let stages = fused_stages(builder.finish(vec![out]).unwrap(), &[values(&[3, 7], 11)], 7);

    assert_eq!(stages, 5);
}

/// Per-channel parameters along axis 1 of three channels.
fn per_channel(dtype: QuantDType, maps: [(f32, i32); 3]) -> QuantParams {
    let maps = maps
        .into_iter()
        .map(|(scale, zero_point)| Affine {
            scale: Scale::new(scale),
            zero_point: ZeroPoint::new(zero_point),
        })
        .collect();
    QuantParams::new(dtype, Granularity::PerAxis { axis: Axis::new(1), maps }).unwrap()
}

#[test]
fn the_tables_of_each_node_keep_their_own_values() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32s(&[4, 3]));
    let onto = per_channel(QuantDType::I8, [(0.5, 1), (1.0, -2), (3.0, 0)]);
    let quantized =
        one(&mut builder, Op::Quantize(QuantizeAttrs::new(onto, Overflow::Saturate)), &[x]);
    let from = per_channel(QuantDType::I8, [(1.0, 0), (0.05, 3), (0.5, -1)]);
    let attrs = DequantizeAttrs { params: from, output: DType::F32 };
    let out = one(&mut builder, Op::Dequantize(attrs), &[quantized]);

    let stages = fused_stages(builder.finish(vec![out]).unwrap(), &[values(&[4, 3], 12)], 1);

    assert_eq!(stages, 1);
}
