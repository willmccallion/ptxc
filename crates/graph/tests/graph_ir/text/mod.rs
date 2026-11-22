//! The text format: exact syntax, names and constants, every op's round
//! trip, and parse errors.

mod cases;
mod errors;
mod round_trip;
mod syntax;

use ptxc_core::HostTensor;
use ptxc_graph::text::{parse, print};
use ptxc_graph::{Graph, GraphBuilder, Op, TensorType};

/// A graph that applies `op` to fresh inputs of `types` and returns every
/// result.
fn single_op(op: Op, types: &[TensorType]) -> Graph {
    let mut builder = GraphBuilder::new();
    let inputs = types.iter().map(|ty| builder.add_input(ty.clone())).collect();
    let outputs = builder.add_node(op, inputs).unwrap();
    builder.finish(outputs).unwrap()
}

/// Parses the printed graph and checks it prints the same, has the same ops
/// in order, and verifies. `elided` supplies large constants.
fn assert_round_trips(graph: &Graph, elided: &[HostTensor]) {
    let text = print(graph);
    let parsed = parse(&text, elided).unwrap_or_else(|error| panic!("{error}\n{text}"));
    assert_eq!(print(&parsed), text);
    assert_eq!(ops_in_order(&parsed), ops_in_order(graph), "{text}");
    assert_eq!(parsed.verify(), Ok(()));
}

fn ops_in_order(graph: &Graph) -> Vec<Op> {
    let order = graph.topological_order().unwrap();
    order.into_iter().map(|id| graph.node(id).unwrap().op().clone()).collect()
}
