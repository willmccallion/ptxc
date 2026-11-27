//! `GreedyEngine`: applies rules in one topological sweep, to a fixed point.

use super::engine::RewriteEngine;
use super::pattern::Bindings;
use super::rule::RewriteRule;
use super::template::Replacement;
use crate::pass::PassError;
use ptxc_graph::{Graph, NodeId, Rewriter, ValueId};

/// Rewrites each node's output with the first rule that applies, as MLIR's
/// greedy pattern rewrite driver does: the output's readers move to the
/// replacement, and the matched nodes left unread are erased.
// NOTE: MLIR's driver keeps a worklist, re-adding the ops a rewrite
// touches, since its rewrites may create ops anywhere. A pattern here
// matches a node by what it reads, and a rewrite only changes the readers
// of the value it replaces, which come after it; so one sweep in
// topological order visits each node after every rewrite that can change
// its match. The nodes a template builds are not visited themselves, though
// a later node's match may read them, as a quantization reads the product
// QDQ folding builds: no rule set yet matches at a node a rule builds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GreedyEngine;

impl RewriteEngine for GreedyEngine {
    fn apply(
        &self,
        graph: &mut Graph,
        rules: &[RewriteRule],
    ) -> Result<Vec<&'static str>, PassError> {
        let order = graph.topological_order()?;
        let mut applied = Vec::new();
        let mut rewriter = graph.rewriter();
        for node in order {
            let Some(Applicable { rule, output, bindings, replacement }) =
                first_applicable(rewriter.graph(), node, rules)
            else {
                continue;
            };
            let replacement = replacement.build(&mut rewriter)?;
            let Some(replacement) = replacement.filter(|&value| value != output) else { continue };
            let matched = rule.lhs.nodes(rewriter.graph(), &bindings);
            rewriter.replace_all_uses(output, replacement)?;
            erase_unread(&mut rewriter, &matched)?;
            applied.push(rule.name);
        }
        if !applied.is_empty() {
            rewriter.compact();
        }
        Ok(applied)
    }
}

/// A rule that applies to a node's output.
struct Applicable<'r> {
    rule: &'r RewriteRule,
    output: ValueId,
    bindings: Bindings,
    replacement: Replacement,
}

/// The first of `rules` that applies to `node`'s only output: its pattern
/// matches, its guard holds and its template resolves.
fn first_applicable<'r>(
    graph: &Graph,
    node: NodeId,
    rules: &'r [RewriteRule],
) -> Option<Applicable<'r>> {
    let mut outputs = graph.node(node)?.outputs().iter();
    let (Some(&output), None) = (outputs.next(), outputs.next()) else { return None };
    rules.iter().find_map(|rule| {
        let bindings = rule.lhs.matches(graph, output)?;
        if !rule.guard.holds(graph, &bindings) {
            return None;
        }
        let replacement = rule.rhs.resolve(graph, &bindings)?;
        Some(Applicable { rule, output, bindings, replacement })
    })
}

/// Erases each of `nodes` none of whose outputs is read any more, in
/// order, so a node goes before the nodes it reads.
fn erase_unread(rewriter: &mut Rewriter<'_>, nodes: &[NodeId]) -> Result<(), PassError> {
    for &node in nodes {
        let Some(contents) = rewriter.graph().node(node) else { continue };
        let unread =
            contents.outputs().iter().all(|&output| rewriter.graph().uses(output).is_empty());
        if unread {
            rewriter.erase_node(node)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::GreedyEngine;
    use crate::rewrite::pattern::{Capture, Pattern};
    use crate::rewrite::rule::{Guard, RewriteRule};
    use crate::rewrite::template::{OpTemplate, Template};
    use crate::rewrite::{RewriteEngine, canonical};
    use ptxc_core::DType;
    use ptxc_core::float::Overflow;
    use ptxc_graph::text::{parse, print};
    use ptxc_graph::{BinaryOp, CastAttrs, Op, UnaryOp};

    #[test]
    fn the_engine_reports_each_rule_it_applies_in_order() {
        let text = "\
graph(%x: f32[2, 3]) {
  %0 = cast<to=f32> %x : f32[2, 3]
  %1 = transpose<permutation=[0, 1]> %0 : f32[2, 3]
  %2 = reshape<shape=[2, 3]> %1 : f32[2, 3]
  %3 = cast<to=f16> %2 : f16[2, 3]
  return %3
}
";
        let mut graph = parse(text, &[]).unwrap();

        let applied = GreedyEngine.apply(&mut graph, &canonical()).unwrap();

        assert_eq!(applied, ["cast-to-own-type", "transpose-in-place", "reshape-to-own-shape"]);
        assert_eq!(graph.nodes().count(), 1);
    }

    /// The rule `name`, replacing `neg x` with `rhs`.
    fn from_negation(name: &'static str, rhs: Template) -> RewriteRule {
        let lhs = Pattern::Op {
            op: "neg",
            operands: vec![Pattern::Any(Capture("x"))],
            capture: Capture("y"),
        };
        RewriteRule { name, lhs, rhs, guard: Guard::All(Vec::new()) }
    }

    fn node(op: Op, operands: Vec<Template>) -> Template {
        Template::Node { op: OpTemplate::Exact(op), operands }
    }

    #[test]
    fn a_rule_building_a_node_its_type_rule_refuses_does_not_apply_and_the_next_rule_does() {
        let text = "graph(%x: f32[3]) {\n  %0 = neg %x : f32[3]\n  return %0\n}\n";
        let mut graph = parse(text, &[]).unwrap();
        let x = Template::Captured(Capture("x"));
        let to_i32 = Op::Cast(CastAttrs::new(DType::I32, Overflow::Saturate));
        let refused =
            node(Op::Binary(BinaryOp::Add), vec![x.clone(), node(to_i32, vec![x.clone()])]);
        let rules = [
            from_negation("refused", refused),
            from_negation("absolute", node(Op::Unary(UnaryOp::Abs), vec![x])),
        ];

        let applied = GreedyEngine.apply(&mut graph, &rules).unwrap();

        assert_eq!(applied, ["absolute"]);
        assert_eq!(print(&graph), "graph(%x: f32[3]) {\n  %0 = abs %x : f32[3]\n  return %0\n}\n");
    }
}
