use crate::{add_one, stages, tensor_type};
use ptxc_core::{Affine, DType, Granularity, QuantDType, QuantParams, Scale, ZeroPoint};
use ptxc_graph::{BinaryOp, GraphBuilder, Op, QMatMulAttrs, UnaryOp};
use ptxc_lower::{Binding, LowerError};

#[test]
fn a_stage_binds_each_graph_value_once() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(tensor_type(DType::F32, &[4]));
    let doubled = add_one(&mut builder, Op::Binary(BinaryOp::Add), &[x, x]);
    let graph = builder.finish(vec![doubled]).unwrap();

    let stages = stages(graph).unwrap();

    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0].inputs().iter().copied().collect::<Vec<_>>(), [Binding::Value(x)]);
    assert_eq!(stages[0].outputs(), [Binding::Value(doubled)]);
    assert_eq!(stages[0].compute().name().as_str(), "add_0");
}

#[test]
fn a_node_whose_output_no_one_reads_has_no_stage() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(tensor_type(DType::F32, &[4]));
    let _unread = add_one(&mut builder, Op::Unary(UnaryOp::Neg), &[x]);
    let out = add_one(&mut builder, Op::Unary(UnaryOp::Abs), &[x]);
    let graph = builder.finish(vec![out]).unwrap();

    let stages = stages(graph).unwrap();

    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0].outputs(), [Binding::Value(out)]);
}

#[test]
fn a_quantized_product_on_fp8_grids_is_reported_with_its_type() {
    let grid = || {
        let affine = Affine { scale: Scale::new(0.5), zero_point: ZeroPoint::new(0) };
        QuantParams::new(QuantDType::F8E4M3Fn, Granularity::PerTensor(affine)).unwrap()
    };
    let mut builder = GraphBuilder::new();
    let lhs = builder.add_input(tensor_type(DType::F8E4M3Fn, &[2, 3]));
    let rhs = builder.add_input(tensor_type(DType::F8E4M3Fn, &[3, 2]));
    let output = {
        let affine = Affine { scale: Scale::new(1.0), zero_point: ZeroPoint::new(0) };
        QuantParams::new(QuantDType::I8, Granularity::PerTensor(affine)).unwrap()
    };
    let attrs = QMatMulAttrs { a: grid(), b: grid(), output };
    let product = add_one(&mut builder, Op::QMatMul(attrs), &[lhs, rhs]);
    let graph = builder.finish(vec![product]).unwrap();

    let error = stages(graph).err();

    let expected = LowerError::UnsupportedDType { op: "qmatmul", dtype: DType::F8E4M3Fn };
    assert_eq!(error, Some(expected));
}
