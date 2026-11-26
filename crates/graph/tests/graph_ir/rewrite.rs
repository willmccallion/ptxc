use crate::common::{add_one, binary, f32_type, unary};
use ptxc_core::HostTensor;
use ptxc_core::Idx;
use ptxc_graph::text::{parse, print};
use ptxc_graph::{
    BinaryOp, Graph, GraphBuilder, GraphOutputIndex, RewriteError, UnaryOp, Use, ValueId,
    ValueName, VerifyError,
};

/// `out = abs(neg(x))`, returning the graph, `x`, the neg output and `out`.
fn chain() -> (Graph, ValueId, ValueId, ValueId) {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let negated = add_one(&mut builder, unary(UnaryOp::Neg), &[x]);
    let out = add_one(&mut builder, unary(UnaryOp::Abs), &[negated]);
    (builder.finish(vec![out]).unwrap(), x, negated, out)
}

#[test]
fn replace_all_uses_redirects_operands_and_graph_outputs() {
    let (mut graph, x, negated, out) = chain();
    let abs = graph.producer(out).unwrap();
    let mut rewriter = graph.rewriter();
    let exp = rewriter.add_node(unary(UnaryOp::Exp), vec![x]).unwrap()[0];

    rewriter.replace_all_uses(negated, exp).unwrap();
    rewriter.replace_all_uses(out, exp).unwrap();

    assert_eq!(graph.verify(), Ok(()));
    assert_eq!(graph.uses(negated), []);
    assert_eq!(graph.uses(out), []);
    assert_eq!(graph.node(abs).unwrap().operands().iter().copied().collect::<Vec<_>>(), [exp]);
    assert_eq!(graph.outputs().iter().copied().collect::<Vec<_>>(), [exp]);
    assert_eq!(graph.uses(exp).len(), 2);
}

#[test]
fn replacing_with_a_different_type_is_rejected_and_changes_nothing() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let wide = builder.add_input(f32_type(&[2, 4]));
    let out = add_one(&mut builder, unary(UnaryOp::Neg), &[x]);
    let mut graph = builder.finish(vec![out]).unwrap();
    let uses_before = graph.uses(x).to_vec();

    let error = graph.rewriter().replace_all_uses(x, wide).unwrap_err();

    assert!(
        matches!(error, RewriteError::TypeMismatch { old, new, .. } if old == x && new == wide)
    );
    assert_eq!(graph.uses(x), uses_before);
    assert_eq!(graph.uses(wide), []);
}

#[test]
fn replacing_a_value_with_itself_changes_nothing() {
    let (mut graph, _, negated, _) = chain();
    let uses_before = graph.uses(negated).to_vec();

    graph.rewriter().replace_all_uses(negated, negated).unwrap();

    assert_eq!(graph.uses(negated), uses_before);
    assert_eq!(graph.verify(), Ok(()));
}

#[test]
fn a_node_whose_output_is_used_cannot_be_erased() {
    let (mut graph, _, negated, _) = chain();
    let neg = graph.producer(negated).unwrap();

    let error = graph.rewriter().erase_node(neg).unwrap_err();

    assert_eq!(error, RewriteError::OutputStillUsed { node: neg, value: negated });
    assert!(graph.node(neg).is_some());
}

#[test]
fn erasing_a_node_removes_it_and_its_operand_uses() {
    let (mut graph, x, negated, out) = chain();
    let neg = graph.producer(negated).unwrap();
    let mut rewriter = graph.rewriter();
    rewriter.replace_all_uses(negated, x).unwrap();

    rewriter.erase_node(neg).unwrap();

    assert!(graph.node(neg).is_none());
    assert!(!graph.is_live(negated));
    assert_eq!(graph.uses(x).len(), 1);
    assert_eq!(graph.verify(), Ok(()));
    assert_eq!(graph.rewriter().erase_node(neg), Err(RewriteError::NodeNotLive(neg)));
    assert!(graph.is_live(out));
}

#[test]
fn an_erased_nodes_output_cannot_be_used() {
    let (mut graph, x, negated, _) = chain();
    let neg = graph.producer(negated).unwrap();
    let mut rewriter = graph.rewriter();
    rewriter.replace_all_uses(negated, x).unwrap();
    rewriter.erase_node(neg).unwrap();

    let error = rewriter.add_node(unary(UnaryOp::Abs), vec![negated]).unwrap_err();

    assert_eq!(error, ptxc_graph::NodeError::NotLive(negated));
}

#[test]
fn compact_renumbers_densely_and_keeps_the_graph_valid() {
    let (mut graph, x, negated, _) = chain();
    let neg = graph.producer(negated).unwrap();
    let mut rewriter = graph.rewriter();
    rewriter.replace_all_uses(negated, x).unwrap();
    rewriter.erase_node(neg).unwrap();

    graph.rewriter().compact();

    assert_eq!(graph.verify(), Ok(()));
    let nodes: Vec<_> = graph.nodes().map(|(id, node)| (id, node.op().clone())).collect();
    assert_eq!(nodes, [(neg, unary(UnaryOp::Abs))]);
    let [output] = graph.outputs().iter().copied().collect::<Vec<_>>()[..] else { panic!() };
    assert_eq!(output, negated, "the abs output takes the freed value id");
    assert_eq!(graph.producer(output), Some(neg));
    assert_eq!(graph.uses(x), [Use::Operand { node: neg, index: ptxc_core::Idx::new(0) }]);
}

#[test]
fn compact_without_erasures_changes_nothing() {
    let (mut graph, x, negated, out) = chain();
    let before: Vec<_> = graph.nodes().map(|(id, node)| (id, node.operands().clone())).collect();

    graph.rewriter().compact();

    let after: Vec<_> = graph.nodes().map(|(id, node)| (id, node.operands().clone())).collect();
    assert_eq!(before, after);
    assert_eq!([x, negated, out].map(|value| graph.is_live(value)), [true; 3]);
    assert_eq!(graph.verify(), Ok(()));
}

/// `x + [3, 4]`, with the constant `[5, 6]` as a second output, and with
/// `unread`, a constant nothing reads.
fn sum_and_constant(unread: Option<HostTensor>) -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[2]));
    if let Some(tensor) = unread {
        let _unread = builder.add_constant(tensor);
    }
    let read = builder.add_constant(HostTensor::vector(vec![3.0_f32, 4.0].into()));
    let output = builder.add_constant(HostTensor::vector(vec![5.0_f32, 6.0].into()));
    let sum = add_one(&mut builder, binary(BinaryOp::Add), &[x, read]);
    builder.finish(vec![sum, output]).unwrap()
}

#[test]
fn compact_drops_constants_nothing_reads_and_keeps_those_read() {
    let mut graph = sum_and_constant(Some(HostTensor::vector(vec![1.0_f32, 2.0].into())));

    graph.rewriter().compact();

    assert_eq!(graph.verify(), Ok(()));
    assert_eq!(print(&graph), print(&sum_and_constant(None)));
}

#[test]
fn a_rewrite_that_closes_a_cycle_is_reported() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[4]));
    let first = add_one(&mut builder, binary(BinaryOp::Add), &[x, x]);
    let second = add_one(&mut builder, unary(UnaryOp::Neg), &[first]);
    let mut graph = builder.finish(vec![second]).unwrap();

    graph.rewriter().replace_all_uses(x, second).unwrap();

    assert!(graph.topological_order().is_err());
    assert!(matches!(graph.verify(), Err(VerifyError::Cycle(_))));
}

#[test]
fn an_output_keeps_its_name_whichever_value_it_is_made_to_read() {
    let text = "\
graph(%x: f32[4]) {
  %y0 = neg %x : f32[4]
  %y1 = neg %x : f32[4]
  return %y0, %y1
}
";
    let mut graph = parse(text, &[]).unwrap();
    let [y0, y1] = graph.outputs().iter().copied().collect::<Vec<_>>()[..] else { panic!() };

    graph.rewriter().replace_all_uses(y1, y0).unwrap();

    let second = GraphOutputIndex::new(1);
    assert_eq!(graph.output_name(second).map(ValueName::as_str), Some("y1"));
    assert_eq!(graph.outputs()[second], y0);
    let printed = print(&graph);
    assert!(printed.ends_with("  return %y0, %y0 as %y1\n}\n"), "{printed}");
    assert_eq!(print(&parse(&printed, &[]).unwrap()), printed);
}

#[test]
fn an_output_can_only_be_named_with_a_name() {
    let text = "graph(%x: f32[4]) {\n  return %x as %0\n}\n";

    let error = parse(text, &[]).unwrap_err();

    assert!(error.to_string().contains("a name such as `%y`"), "{error}");
}
