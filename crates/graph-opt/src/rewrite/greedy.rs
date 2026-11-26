//! `GreedyEngine`: applies rules in one topological sweep, to a fixed point.

use super::engine::RewriteEngine;
use super::pattern::Bindings;
use super::rule::RewriteRule;
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
// its match. The nodes a template builds are not visited: no rule set yet
// matches what another rule builds.
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
            let Some((rule, output, bindings)) = first_match(rewriter.graph(), node, rules) else {
                continue;
            };
            let replacement = rule.rhs.instantiate(&mut rewriter, &bindings)?;
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

/// The first of `rules` whose pattern matches `node`'s only output and
/// whose guard holds, with that output and the match's bindings.
fn first_match<'r>(
    graph: &Graph,
    node: NodeId,
    rules: &'r [RewriteRule],
) -> Option<(&'r RewriteRule, ValueId, Bindings)> {
    let mut outputs = graph.node(node)?.outputs().iter();
    let (Some(&output), None) = (outputs.next(), outputs.next()) else { return None };
    rules.iter().find_map(|rule| {
        let bindings = rule.lhs.matches(graph, output)?;
        rule.guard.holds(graph, &bindings).then_some((rule, output, bindings))
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
