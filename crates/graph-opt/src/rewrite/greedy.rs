//! `GreedyEngine`: applies rules in one topological sweep, to a fixed point.

use super::engine::RewriteEngine;
use super::rule::RewriteRule;
use crate::pass::PassError;
use ptxc_graph::{Graph, NodeId, ValueId};

/// Rewrites each node's output with the first rule that applies, as MLIR's
/// greedy pattern rewrite driver does, moving the output's readers to the
/// replacement and erasing the node.
// NOTE: MLIR's driver keeps a worklist, re-adding the ops a rewrite
// touches, since its rewrites may create ops anywhere. A pattern here
// matches a node by what it reads, and a rewrite only changes the readers
// of the value it replaces, which come after it; so one sweep in
// topological order visits each node after every rewrite that can change
// its match, and ends at a fixed point. Templates that add nodes would
// need them visited too.
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
            let Some(rewrite) = first_rewrite(rewriter.graph(), node, rules) else { continue };
            rewriter.replace_all_uses(rewrite.output, rewrite.replacement)?;
            rewriter.erase_node(node)?;
            applied.push(rewrite.rule);
        }
        if !applied.is_empty() {
            rewriter.compact();
        }
        Ok(applied)
    }
}

/// A rule's rewrite of a node's output.
struct Rewrite {
    /// The rule's name.
    rule: &'static str,
    /// The value replaced.
    output: ValueId,
    /// The value its readers read instead.
    replacement: ValueId,
}

/// The rewrite of `node`'s only output by the first of `rules` that
/// matches it, whose guard holds and whose template gives another value.
fn first_rewrite(graph: &Graph, node: NodeId, rules: &[RewriteRule]) -> Option<Rewrite> {
    let mut outputs = graph.node(node)?.outputs().iter();
    let (Some(&output), None) = (outputs.next(), outputs.next()) else { return None };
    rules.iter().find_map(|rule| {
        let bindings = rule.lhs.matches(graph, output)?;
        if !rule.guard.holds(graph, &bindings) {
            return None;
        }
        let replacement = rule.rhs.instantiate(&bindings).filter(|&value| value != output)?;
        Some(Rewrite { rule: rule.name, output, replacement })
    })
}

#[cfg(test)]
mod tests {
    use super::GreedyEngine;
    use crate::rewrite::{RewriteEngine, canonical};
    use ptxc_graph::text::parse;

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
}
