//! `Pattern`: the left-hand side of a rule, matched against the graph.

use ptxc_graph::{Graph, ValueId};
use std::collections::HashMap;

/// A name a pattern binds a value to, for the guard and template to read.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Capture(pub(crate) &'static str);

/// A tree of ops over captured values, as MLIR's declarative rewrite
/// patterns are.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Pattern {
    /// Any value.
    Any(Capture),
    /// The only output of a node whose op has the mnemonic `op`, reading
    /// operands that match `operands` in order.
    Op {
        /// The op's mnemonic in the text format, such as `reshape`.
        op: &'static str,
        /// One pattern per operand.
        operands: Vec<Self>,
        /// The name the output is bound to.
        capture: Capture,
    },
}

/// The values a match bound, by capture.
pub(crate) type Bindings = HashMap<Capture, ValueId>;

impl Pattern {
    /// The bindings under which `self` matches `value` in `graph`, if it
    /// does. A capture appearing twice must bind the same value both times.
    pub(crate) fn matches(&self, graph: &Graph, value: ValueId) -> Option<Bindings> {
        let mut bindings = Bindings::new();
        self.bind(graph, value, &mut bindings).then_some(bindings)
    }

    fn bind(&self, graph: &Graph, value: ValueId, bindings: &mut Bindings) -> bool {
        match self {
            Self::Any(capture) => bound(bindings, *capture, value),
            Self::Op { op, operands, capture } => {
                let Some(node) = graph.producer(value).and_then(|node| graph.node(node)) else {
                    return false;
                };
                let single_output = node.outputs().len() == 1;
                if !single_output || node.op().mnemonic() != *op {
                    return false;
                }
                operands.len() == node.operands().len()
                    && operands
                        .iter()
                        .zip(node.operands().iter())
                        .all(|(pattern, &operand)| pattern.bind(graph, operand, bindings))
                    && bound(bindings, *capture, value)
            }
        }
    }
}

/// Binds `capture` to `value` unless it is bound to another value already.
fn bound(bindings: &mut Bindings, capture: Capture, value: ValueId) -> bool {
    *bindings.entry(capture).or_insert(value) == value
}

#[cfg(test)]
mod tests {
    use super::{Capture, Pattern};
    use ptxc_graph::text::parse;
    use ptxc_graph::{Graph, ValueId};

    fn graph(text: &str) -> Graph {
        parse(text, &[]).unwrap()
    }

    fn output(graph: &Graph) -> ValueId {
        graph.outputs().iter().copied().next().unwrap()
    }

    fn any(name: &'static str) -> Pattern {
        Pattern::Any(Capture(name))
    }

    fn op(op: &'static str, operands: Vec<Pattern>, capture: &'static str) -> Pattern {
        Pattern::Op { op, operands, capture: Capture(capture) }
    }

    #[test]
    fn a_nested_pattern_binds_every_value_it_names() {
        let graph = graph(
            "graph(%x: f32[4]) {\n  %0 = neg %x : f32[4]\n  %1 = abs %0 : f32[4]\n  return %1\n}\n",
        );
        let pattern = op("abs", vec![op("neg", vec![any("x")], "negated")], "y");

        let bindings = pattern.matches(&graph, output(&graph)).unwrap();

        assert_eq!(bindings[&Capture("x")], graph.inputs().iter().copied().next().unwrap());
        assert_eq!(bindings[&Capture("y")], output(&graph));
        assert_eq!(bindings.len(), 3);
        assert!(
            op("abs", vec![op("exp", vec![any("x")], "e")], "y")
                .matches(&graph, output(&graph))
                .is_none()
        );
    }

    #[test]
    fn a_capture_named_twice_must_bind_one_value() {
        let same = graph("graph(%a: f32[4]) {\n  %0 = add %a, %a : f32[4]\n  return %0\n}\n");
        let different =
            graph("graph(%a: f32[4], %b: f32[4]) {\n  %0 = add %a, %b : f32[4]\n  return %0\n}\n");
        let doubled = op("add", vec![any("x"), any("x")], "y");

        assert!(doubled.matches(&same, output(&same)).is_some());
        assert!(doubled.matches(&different, output(&different)).is_none());
    }

    #[test]
    fn an_op_matches_only_with_one_output_and_as_many_operands() {
        let split = graph(
            "graph(%x: f32[4]) {\n  %0, %1 = split<axis=0, sizes=[2, 2]> %x : f32[2], f32[2]\n  return %0\n}\n",
        );
        let negated = graph("graph(%x: f32[4]) {\n  %0 = neg %x : f32[4]\n  return %0\n}\n");

        assert!(op("split", vec![any("x")], "y").matches(&split, output(&split)).is_none());
        assert!(op("neg", vec![], "y").matches(&negated, output(&negated)).is_none());
        assert!(
            op("neg", vec![any("x"), any("z")], "y").matches(&negated, output(&negated)).is_none()
        );
        assert!(op("neg", vec![any("x")], "y").matches(&negated, output(&negated)).is_some());
    }
}
