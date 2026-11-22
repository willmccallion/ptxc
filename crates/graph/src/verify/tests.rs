//! Corrupted graphs the verifier must reject. A corrupt graph cannot be built
//! through the public API, so these tests edit the arenas directly.

use super::VerifyError;
use crate::builder::GraphBuilder;
use crate::graph::Graph;
use crate::node::{NodeId, NodeSlot, OperandIndex};
use crate::ops::{BinaryOp, Op, UnaryOp};
use crate::tensor_type::{Layout, TensorType};
use crate::value::{GraphInputIndex, Use, ValueDef, ValueId};
use ptxc_core::{DType, Dim, HostTensor, Idx, Shape};

fn f32_type(dims: &[u64]) -> TensorType {
    let shape = Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap();
    TensorType::new(DType::F32, shape, Layout::RowMajor)
}

/// `out = neg(x + c)` with `x: f32[2, 3]` and constant `c: f32[3]`.
struct Fixture {
    graph: Graph,
    x: ValueId,
    c: ValueId,
    sum: ValueId,
    out: ValueId,
    add: NodeId,
    neg: NodeId,
}

fn fixture() -> Fixture {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[2, 3]));
    let shape = Shape::new(vec![Dim::new(3)]).unwrap();
    let c = builder.add_constant(HostTensor::new(shape, vec![1.0_f32; 3].into()).unwrap());
    let [sum] = builder.add_node(Op::Binary(BinaryOp::Add), vec![x, c]).unwrap()[..] else {
        panic!("add has one output")
    };
    let [out] = builder.add_node(Op::Unary(UnaryOp::Neg), vec![sum]).unwrap()[..] else {
        panic!("neg has one output")
    };
    let graph = builder.finish(vec![out]).unwrap();
    let add = graph.producer(sum).unwrap();
    let neg = graph.producer(out).unwrap();
    Fixture { graph, x, c, sum, out, add, neg }
}

fn node_mut(graph: &mut Graph, id: NodeId) -> &mut crate::node::Node {
    let NodeSlot::Live(node) = &mut graph.nodes[id] else { panic!("node {id:?} is erased") };
    node
}

#[test]
fn well_formed_graph_verifies() {
    assert_eq!(fixture().graph.verify(), Ok(()));
}

#[test]
fn reading_an_erased_nodes_output_is_rejected() {
    let Fixture { mut graph, sum, neg, add, .. } = fixture();
    graph.nodes[add] = NodeSlot::Erased;

    assert_eq!(graph.verify(), Err(VerifyError::DeadOperand { node: neg, value: sum }));
}

#[test]
fn a_value_whose_defining_slot_holds_another_value_is_rejected() {
    let Fixture { mut graph, sum, out, neg, .. } = fixture();
    node_mut(&mut graph, neg).outputs[crate::node::NodeOutputIndex::new(0)] = sum;

    assert_eq!(graph.verify(), Err(VerifyError::Definition(out)));
}

#[test]
fn a_graph_input_listed_twice_is_rejected() {
    let Fixture { mut graph, x, .. } = fixture();
    let _: GraphInputIndex = graph.inputs.push(x);

    assert_eq!(graph.verify(), Err(VerifyError::Definition(x)));
}

#[test]
fn an_input_whose_definition_points_elsewhere_is_rejected() {
    let Fixture { mut graph, x, .. } = fixture();
    graph.values[x].def = ValueDef::Input(GraphInputIndex::new(5));

    assert_eq!(graph.verify(), Err(VerifyError::Definition(x)));
}

#[test]
fn a_constant_value_with_the_wrong_type_is_rejected() {
    let Fixture { mut graph, c, .. } = fixture();
    graph.values[c].ty = f32_type(&[4]);

    assert_eq!(graph.verify(), Err(VerifyError::ConstantType(c)));
}

#[test]
fn a_stale_use_list_is_rejected() {
    let Fixture { mut graph, x, add, .. } = fixture();
    graph.uses[x].clear();

    let expected = VerifyError::UseList {
        value: x,
        listed: Vec::new(),
        actual: vec![Use::Operand { node: add, index: OperandIndex::new(0) }],
    };
    assert_eq!(graph.verify(), Err(expected));
}

#[test]
fn a_dead_graph_output_is_rejected() {
    let Fixture { mut graph, out, neg, .. } = fixture();
    graph.nodes[neg] = NodeSlot::Erased;

    assert_eq!(graph.verify(), Err(VerifyError::DeadGraphOutput(out)));
}

#[test]
fn a_cycle_is_rejected() {
    let Fixture { mut graph, x, out, add, .. } = fixture();
    let first = OperandIndex::new(0);
    node_mut(&mut graph, add).operands[first] = out;
    graph.uses[x].clear();
    graph.uses[out].push(Use::Operand { node: add, index: first });

    assert!(matches!(graph.verify(), Err(VerifyError::Cycle(_))), "{:?}", graph.verify());
}

#[test]
fn output_types_that_differ_from_the_type_rule_are_rejected() {
    let Fixture { mut graph, out, neg, .. } = fixture();
    graph.values[out].ty = f32_type(&[3]);

    let expected = VerifyError::OutputTypes {
        node: neg,
        recorded: vec![f32_type(&[3])],
        inferred: vec![f32_type(&[2, 3])],
    };
    assert_eq!(graph.verify(), Err(expected));
}

#[test]
fn an_op_that_rejects_its_operands_is_rejected() {
    let Fixture { mut graph, neg, .. } = fixture();
    node_mut(&mut graph, neg).op = Op::Unary(UnaryOp::Not);

    assert!(
        matches!(graph.verify(), Err(VerifyError::TypeRule { node, .. }) if node == neg),
        "{:?}",
        graph.verify()
    );
}
