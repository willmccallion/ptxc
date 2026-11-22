use crate::common::{add_one, binary, f32_type, unary};
use ptxc_graph::{BinaryOp, GraphBuilder, NodeId, UnaryOp};

#[test]
fn a_built_graph_comes_out_in_definition_order() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let left = add_one(&mut builder, unary(UnaryOp::Neg), &[x]);
    let right = add_one(&mut builder, unary(UnaryOp::Abs), &[x]);
    let joined = add_one(&mut builder, binary(BinaryOp::Add), &[left, right]);
    let graph = builder.finish(vec![joined]).unwrap();
    let definition_order: Vec<NodeId> =
        [left, right, joined].map(|value| graph.producer(value).unwrap()).into();

    assert_eq!(graph.topological_order(), Ok(definition_order));
}

#[test]
fn a_node_added_later_comes_before_the_nodes_that_read_it() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let negated = add_one(&mut builder, unary(UnaryOp::Neg), &[x]);
    let out = add_one(&mut builder, unary(UnaryOp::Abs), &[negated]);
    let mut graph = builder.finish(vec![out]).unwrap();
    let mut rewriter = graph.rewriter();
    let exp = rewriter.add_node(unary(UnaryOp::Exp), vec![x]).unwrap()[0];
    rewriter.replace_all_uses(negated, exp).unwrap();

    let order = graph.topological_order().unwrap();

    let position = |value| {
        let node = graph.producer(value).unwrap();
        order.iter().position(|&id| id == node).unwrap()
    };
    assert!(position(exp) < position(out));
    assert_eq!(order.len(), 3);
}

#[test]
fn erased_nodes_are_left_out() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let negated = add_one(&mut builder, unary(UnaryOp::Neg), &[x]);
    let mut graph = builder.finish(vec![x]).unwrap();
    let neg = graph.producer(negated).unwrap();
    graph.rewriter().erase_node(neg).unwrap();

    assert_eq!(graph.topological_order(), Ok(Vec::new()));
}

#[test]
fn a_graph_without_nodes_has_an_empty_order() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let graph = builder.finish(vec![x]).unwrap();

    assert_eq!(graph.topological_order(), Ok(Vec::new()));
}
