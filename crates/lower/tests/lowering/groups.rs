use crate::{add_one, stages, tensor_type};
use ptxc_core::DType;
use ptxc_graph::{BinaryOp, GlobalPoolKind, GraphBuilder, Op, UnaryOp};
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
fn an_op_without_a_lowering_is_reported_by_name() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(tensor_type(DType::F32, &[1, 2, 3, 3]));
    let pooled = add_one(&mut builder, Op::GlobalPool(GlobalPoolKind::Average), &[x]);
    let graph = builder.finish(vec![pooled]).unwrap();

    let error = stages(graph).err();

    assert_eq!(error, Some(LowerError::Unsupported { op: "global_average_pool" }));
}
