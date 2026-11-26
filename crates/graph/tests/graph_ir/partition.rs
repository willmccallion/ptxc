use crate::common::{add_one, binary, f32_type, unary};
use ptxc_core::{Dim, Idx, Shape};
use ptxc_graph::{
    BinaryOp, Graph, GraphBuilder, GroupId, GroupKind, NodeId, NodeOutputIndex, Op, PartitionError,
    Partitioned, UnaryOp, ValueId, VariadicOp,
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

/// `out = exp(abs(neg(x)))` and `x`, returning the graph, `x` and the
/// three nodes in order.
fn chain() -> (Graph, ValueId, [NodeId; 3]) {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let negated = add_one(&mut builder, unary(UnaryOp::Neg), &[x]);
    let absolute = add_one(&mut builder, unary(UnaryOp::Abs), &[negated]);
    let out = add_one(&mut builder, unary(UnaryOp::Exp), &[absolute]);
    let graph = builder.finish(vec![out]).unwrap();
    let nodes = [negated, absolute, out].map(|value| graph.producer(value).unwrap());
    (graph, x, nodes)
}

#[test]
fn groups_are_numbered_after_the_groups_they_read() {
    let (graph, x, [neg, abs, exp]) = chain();
    let absolute = graph.node(abs).unwrap().outputs()[NodeOutputIndex::new(0)];

    let partitioned = Partitioned::from_groups(graph, &[vec![exp], vec![abs, neg]]).unwrap();

    let groups = partitioned.groups();
    assert_eq!(groups[GroupId::new(0)].nodes(), [neg, abs]);
    assert_eq!(groups[GroupId::new(0)].inputs(), [x]);
    assert_eq!(groups[GroupId::new(0)].outputs(), [absolute]);
    assert_eq!(groups[GroupId::new(0)].kind(), GroupKind::Injective);
    assert_eq!(groups[GroupId::new(1)].nodes(), [exp]);
    assert_eq!(groups[GroupId::new(1)].inputs(), [absolute]);
    assert_eq!(
        [neg, abs, exp].map(|node| partitioned.group_of(node)),
        [0, 0, 1].map(|g| Some(GroupId::new(g)))
    );
}

#[test]
fn a_value_read_inside_and_outside_its_group_is_one_of_its_outputs() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let negated = add_one(&mut builder, unary(UnaryOp::Neg), &[x]);
    let absolute = add_one(&mut builder, unary(UnaryOp::Abs), &[negated]);
    let graph = builder.finish(vec![absolute, negated]).unwrap();
    let nodes = [negated, absolute].map(|value| graph.producer(value).unwrap()).to_vec();

    let partitioned = Partitioned::from_groups(graph, &[nodes]).unwrap();

    assert_eq!(partitioned.groups()[GroupId::new(0)].outputs(), [negated, absolute]);
}

#[test]
fn a_product_with_its_epilogue_is_a_product_group() {
    let mut builder = GraphBuilder::new();
    let lhs = builder.add_input(f32_type(&[2, 3]));
    let rhs = builder.add_input(f32_type(&[3, 2]));
    let bias = builder.add_input(f32_type(&[2]));
    let product = add_one(&mut builder, Op::MatMul, &[lhs, rhs]);
    let biased = add_one(&mut builder, binary(BinaryOp::Add), &[product, bias]);
    let graph = builder.finish(vec![biased]).unwrap();
    let nodes = [product, biased].map(|value| graph.producer(value).unwrap()).to_vec();

    let partitioned = Partitioned::from_groups(graph, &[nodes]).unwrap();

    assert_eq!(partitioned.groups()[GroupId::new(0)].kind(), GroupKind::MatMulEpilogue);
    assert_eq!(partitioned.groups()[GroupId::new(0)].inputs(), [lhs, rhs, bias]);
}

#[test]
fn groups_that_do_not_partition_the_graph_are_refused() {
    let (graph, _, [neg, abs, exp]) = chain();
    let refused =
        |groups: &[Vec<NodeId>]| Partitioned::from_groups(graph.clone(), groups).unwrap_err();

    assert_eq!(refused(&[vec![neg, abs], vec![abs, exp]]), PartitionError::Repeated(abs));
    assert_eq!(refused(&[vec![neg, exp]]), PartitionError::Missing(abs));
    assert_eq!(
        refused(&[vec![neg, abs, exp, NodeId::new(9)]]),
        PartitionError::NotLive(NodeId::new(9))
    );
    assert_eq!(refused(&[vec![neg, exp], vec![abs]]), PartitionError::GroupCycle);
}

#[test]
fn a_group_of_views_is_a_view() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[2, 6]));
    let flat = Shape::new(vec![Dim::new(12)]).unwrap();
    let square = Shape::new(vec![Dim::new(3), Dim::new(4)]).unwrap();
    let flattened = add_one(&mut builder, Op::Reshape(flat), &[x]);
    let reshaped = add_one(&mut builder, Op::Reshape(square), &[flattened]);
    let graph = builder.finish(vec![reshaped]).unwrap();
    let nodes = [flattened, reshaped].map(|value| graph.producer(value).unwrap()).to_vec();

    let partitioned = Partitioned::from_groups(graph, &[nodes]).unwrap();

    assert_eq!(partitioned.groups()[GroupId::new(0)].kind(), GroupKind::View);
}
