use crate::common::{add_one, binary, f32_constant, f32_type, tensor_type, unary};
use ptxc_core::{DType, Idx};
use ptxc_graph::{
    BinaryOp, GraphBuilder, NodeError, OperandIndex, TypeError, UnaryOp, Use, ValueDef, ValueId,
    ValueName,
};

#[test]
fn built_graph_has_inferred_types_and_verifies() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[2, 3]));
    let c = builder.add_constant(f32_constant(&[3]));
    let sum = add_one(&mut builder, binary(BinaryOp::Add), &[x, c]);
    let mask = add_one(&mut builder, binary(BinaryOp::Greater), &[sum, c]);

    let graph = builder.finish(vec![sum, mask]).unwrap();

    assert_eq!(graph.verify(), Ok(()));
    assert_eq!(graph.value(sum).ty(), &f32_type(&[2, 3]));
    assert_eq!(graph.value(mask).ty(), &tensor_type(DType::Bool, &[2, 3]));
    assert_eq!(graph.value(c).ty(), &f32_type(&[3]));
    assert_eq!(graph.outputs().iter().copied().collect::<Vec<_>>(), [sum, mask]);
    assert_eq!(graph.inputs().iter().copied().collect::<Vec<_>>(), [x]);
}

#[test]
fn constant_values_refer_to_their_tensor() {
    let mut builder = GraphBuilder::new();
    let c = builder.add_constant(f32_constant(&[2, 2]));
    let graph = builder.finish(vec![c]).unwrap();

    let ValueDef::Constant(id) = graph.value(c).def() else { panic!("not a constant") };
    assert_eq!(graph.constant(id).shape(), f32_type(&[2, 2]).shape());
}

#[test]
fn values_are_listed_in_id_order_with_their_definitions() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[3]));
    let c = builder.add_constant(f32_constant(&[3]));
    let sum = add_one(&mut builder, binary(BinaryOp::Add), &[x, c]);
    let graph = builder.finish(vec![sum]).unwrap();

    let values: Vec<(ValueId, bool)> = graph
        .values()
        .map(|(id, value)| (id, matches!(value.def(), ValueDef::Constant(_))))
        .collect();

    assert_eq!(values, [(x, false), (c, true), (sum, false)]);
}

#[test]
fn reading_a_value_twice_records_two_uses() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let doubled = add_one(&mut builder, binary(BinaryOp::Add), &[x, x]);
    let graph = builder.finish(vec![doubled]).unwrap();
    let node = graph.producer(doubled).unwrap();

    let uses = graph.uses(x);

    assert_eq!(
        uses,
        [
            Use::Operand { node, index: OperandIndex::new(0) },
            Use::Operand { node, index: OperandIndex::new(1) }
        ]
    );
}

#[test]
fn graph_outputs_are_uses() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let graph = builder.finish(vec![x, x]).unwrap();

    assert_eq!(graph.uses(x).len(), 2);
    assert!(graph.uses(x).iter().all(|reader| matches!(reader, Use::GraphOutput(_))));
}

#[test]
fn a_type_error_names_the_op() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));

    let error = builder.add_node(unary(UnaryOp::Not), vec![x]).unwrap_err();

    let expected = NodeError::Type {
        op: Box::new(unary(UnaryOp::Not)),
        source: TypeError::UnsupportedDType(DType::F32),
    };
    assert_eq!(error, expected);
}

#[test]
fn an_operand_from_outside_the_graph_is_rejected() {
    let mut builder = GraphBuilder::new();
    let _ = builder.add_input(f32_type(&[4]));
    let foreign = ValueId::new(7);

    let error = builder.add_node(unary(UnaryOp::Neg), vec![foreign]).unwrap_err();

    assert_eq!(error, NodeError::NotLive(foreign));
}

#[test]
fn an_output_from_outside_the_graph_is_rejected() {
    let builder = GraphBuilder::new();

    let error = builder.finish(vec![ValueId::new(0)]).unwrap_err();

    assert_eq!(error, NodeError::NotLive(ValueId::new(0)));
}

#[test]
fn names_are_kept() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    builder.set_name(x, ValueName::new("pixels".to_owned()));
    let graph = builder.finish(vec![x]).unwrap();

    assert_eq!(graph.value(x).name().map(ValueName::as_str), Some("pixels"));
}
