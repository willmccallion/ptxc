//! Graphs written as ONNX models and read back by the importer.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

use ptxc_core::{Axis, DType, Dim, Shape};
use ptxc_graph::{
    Activation, BinaryOp, Graph, GraphBuilder, KeepDims, Layout, Op, ReduceAttrs, ReduceOp,
    SoftmaxAttrs, SoftmaxKind, TensorType, TransposeAttrs, UnaryOp, ValueId, ValueName, VariadicOp,
};
use ptxc_onnx::{ExportError, ImportOptions, import_model, model_to_bytes};

fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap()
}

fn input(builder: &mut GraphBuilder, name: &str, dtype: DType, dims: &[u64]) -> ValueId {
    let value = builder.add_input(TensorType::new(dtype, shape(dims), Layout::RowMajor));
    builder.set_name(value, ValueName::new(name.to_owned()));
    value
}

fn reduce(op: ReduceOp, axes: &[usize], keep_dims: KeepDims) -> Op {
    Op::Reduce(ReduceAttrs { op, axes: axes.iter().copied().map(Axis::new).collect(), keep_dims })
}

/// A chain through every op the exporter maps, each output feeding the
/// next node, and the last one the graph's output.
fn every_exported_op() -> Graph {
    let mut b = GraphBuilder::new();
    let x = input(&mut b, "x", DType::F32, &[2, 3]);
    let w = input(&mut b, "w", DType::F32, &[3, 4]);
    let bias = input(&mut b, "bias", DType::F32, &[4]);
    let empty = input(&mut b, "empty", DType::F32, &[0, 4]);
    let node = |b: &mut GraphBuilder, op: Op, operands: Vec<ValueId>| -> ValueId {
        b.add_node(op, operands).unwrap()[0]
    };
    let mut v = node(&mut b, Op::MatMul, vec![x, w]);
    for op in [
        Op::Unary(UnaryOp::Abs),
        Op::Unary(UnaryOp::Neg),
        Op::Activation(Activation::Relu),
        Op::Activation(Activation::Sigmoid),
        Op::Activation(Activation::Tanh),
        Op::Activation(Activation::Softsign),
        Op::Softmax(SoftmaxAttrs { kind: SoftmaxKind::Softmax, axis: Axis::new(1) }),
        Op::Softmax(SoftmaxAttrs { kind: SoftmaxKind::LogSoftmax, axis: Axis::new(0) }),
    ] {
        v = node(&mut b, op, vec![v]);
    }
    for op in [
        Op::Binary(BinaryOp::Add),
        Op::Binary(BinaryOp::Sub),
        Op::Binary(BinaryOp::Mul),
        Op::Variadic(VariadicOp::Max),
        Op::Variadic(VariadicOp::Min),
    ] {
        v = node(&mut b, op, vec![v, bias]);
    }
    v = node(&mut b, Op::Concat(Axis::new(0)), vec![v, empty]);
    v = node(
        &mut b,
        Op::Transpose(TransposeAttrs { permutation: vec![Axis::new(1), Axis::new(0)] }),
        vec![v],
    );
    v = node(&mut b, Op::Reshape(shape(&[8, 1])), vec![v]);
    v = node(&mut b, reduce(ReduceOp::Sum, &[1], KeepDims::Keep), vec![v]);
    v = node(&mut b, reduce(ReduceOp::Mean, &[0], KeepDims::Keep), vec![v]);
    v = node(&mut b, reduce(ReduceOp::Max, &[0, 1], KeepDims::Drop), vec![v]);
    v = node(&mut b, Op::Reshape(shape(&[1, 1])), vec![v]);
    v = node(&mut b, reduce(ReduceOp::Min, &[1], KeepDims::Drop), vec![v]);
    b.set_name(v, ValueName::new("y".to_owned()));
    b.finish(vec![v]).unwrap()
}

/// Each node's op and its outputs' types, in order.
fn nodes(graph: &Graph) -> Vec<(Op, Vec<TensorType>)> {
    graph
        .nodes()
        .map(|(_, node)| {
            let types = node.outputs().iter().map(|&value| graph.value(value).ty().clone());
            (node.op().clone(), types.collect())
        })
        .collect()
}

fn types<'g>(graph: &Graph, values: impl Iterator<Item = &'g ValueId>) -> Vec<TensorType> {
    values.map(|&value| graph.value(value).ty().clone()).collect()
}

#[test]
fn a_graph_of_every_exported_op_reads_back_as_the_same_graph() {
    let graph = every_exported_op();

    let bytes = model_to_bytes(&graph).unwrap();
    let imported = import_model(&bytes, &ImportOptions::default()).unwrap();

    assert_eq!(nodes(&imported), nodes(&graph));
    assert_eq!(types(&imported, imported.inputs().iter()), types(&graph, graph.inputs().iter()));
    assert_eq!(types(&imported, imported.outputs().iter()), types(&graph, graph.outputs().iter()));
}

#[test]
fn a_zero_extent_survives_a_reshape() {
    let mut b = GraphBuilder::new();
    let x = input(&mut b, "x", DType::F32, &[0, 6]);
    let y = b.add_node(Op::Reshape(shape(&[3, 0, 2])), vec![x]).unwrap()[0];
    let graph = b.finish(vec![y]).unwrap();

    let imported =
        import_model(&model_to_bytes(&graph).unwrap(), &ImportOptions::default()).unwrap();

    assert_eq!(nodes(&imported), nodes(&graph));
}

#[test]
fn an_op_without_an_export_is_an_error_naming_it() {
    let mut b = GraphBuilder::new();
    let x = input(&mut b, "x", DType::F32, &[2]);
    let y = b.add_node(Op::Unary(UnaryOp::Floor), vec![x]).unwrap()[0];
    let graph = b.finish(vec![y]).unwrap();

    let error = model_to_bytes(&graph).unwrap_err();

    assert_eq!(error, ExportError::Op("Unary(Floor)".to_owned()));
}
