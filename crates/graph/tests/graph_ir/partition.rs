use crate::common::{add_one, binary, f32_type, unary};
use ptxc_core::Idx;
use ptxc_graph::{
    BinaryOp, GraphBuilder, GroupId, GroupKind, NodeId, Op, Partitioned, UnaryOp, VariadicOp,
};

#[test]
fn each_node_is_its_own_group_in_topological_order() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let negated = add_one(&mut builder, unary(UnaryOp::Neg), &[x]);
    let out = add_one(&mut builder, unary(UnaryOp::Abs), &[negated]);
    let mut graph = builder.finish(vec![out]).unwrap();
    let mut rewriter = graph.rewriter();
    let exp = rewriter.add_node(unary(UnaryOp::Exp), vec![x]).unwrap()[0];
    rewriter.replace_all_uses(negated, exp).unwrap();
    let neg = graph.producer(negated).unwrap();
    graph.rewriter().erase_node(neg).unwrap();

    let partitioned = Partitioned::one_node_per_group(graph).unwrap();

    let graph = partitioned.graph();
    let nodes: Vec<NodeId> =
        partitioned.groups().iter().flat_map(|group| group.nodes().to_vec()).collect();
    assert_eq!(nodes, [exp, out].map(|value| graph.producer(value).unwrap()));
    for (id, group) in partitioned.groups().iter_enumerated() {
        assert_eq!(partitioned.group_of(group.nodes()[0]), Some(id));
    }
    assert_eq!(partitioned.group_of(neg), None);
}

#[test]
fn a_group_reads_each_input_once_and_exports_only_what_is_read_outside() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let doubled = add_one(&mut builder, binary(BinaryOp::Add), &[x, x]);
    let _unread = add_one(&mut builder, unary(UnaryOp::Neg), &[doubled]);
    let summed = add_one(&mut builder, Op::Variadic(VariadicOp::Sum), &[doubled, x, doubled]);
    let graph = builder.finish(vec![summed]).unwrap();

    let partitioned = Partitioned::one_node_per_group(graph).unwrap();

    let groups = partitioned.groups();
    assert_eq!(groups[GroupId::new(0)].inputs(), [x]);
    assert_eq!(groups[GroupId::new(0)].outputs(), [doubled]);
    assert_eq!(groups[GroupId::new(1)].outputs(), []);
    assert_eq!(groups[GroupId::new(2)].inputs(), [doubled, x]);
    assert_eq!(groups[GroupId::new(2)].outputs(), [summed]);
}

#[test]
fn a_singleton_groups_kind_follows_its_op() {
    let mut builder = GraphBuilder::new();
    let lhs = builder.add_input(f32_type(&[2, 3]));
    let rhs = builder.add_input(f32_type(&[3, 2]));
    let y = add_one(&mut builder, Op::MatMul, &[lhs, rhs]);
    let graph = builder.finish(vec![y]).unwrap();

    let partitioned = Partitioned::one_node_per_group(graph).unwrap();

    assert_eq!(partitioned.groups()[GroupId::new(0)].kind(), GroupKind::MatMulEpilogue);
    assert_eq!(GroupKind::of(&unary(UnaryOp::Exp)), GroupKind::Injective);
    assert_eq!(GroupKind::of(&Op::Reshape(crate::common::shape(&[6]))), GroupKind::View);
}

#[test]
fn an_op_is_named_by_its_text_mnemonic() {
    assert_eq!(binary(BinaryOp::Add).mnemonic(), "add");
    assert_eq!(Op::Where.mnemonic(), "where");
}
