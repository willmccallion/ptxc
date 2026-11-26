//! `RewriteRule`: a named pattern, template and guard, held as data.

use super::pattern::{Bindings, Capture, Pattern};
use super::template::Template;
use ptxc_graph::{Graph, Op, ValueId};

/// A rewrite: where `lhs` matches and `guard` holds, the matched value's
/// readers read `rhs` instead.
///
/// Rules are data rather than closures, so another engine, such as an
/// e-graph's, can apply the same rule sets.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct RewriteRule {
    /// The rule's name, which the engine reports when it applies.
    pub(crate) name: &'static str,
    /// What it matches.
    pub(crate) lhs: Pattern,
    /// What the match is replaced with.
    pub(crate) rhs: Template,
    /// What must hold of the match besides its shape.
    pub(crate) guard: Guard,
}

/// A condition on a match's values, beyond the ops the pattern names.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Guard {
    /// The two values have the same type.
    SameType(Capture, Capture),
    /// The value is a transpose that leaves every axis in place.
    IdentityTranspose(Capture),
}

impl Guard {
    /// Whether the guard holds of `bindings` in `graph`. A capture the
    /// pattern did not bind fails it.
    pub(crate) fn holds(&self, graph: &Graph, bindings: &Bindings) -> bool {
        let value = |capture: &Capture| bindings.get(capture).copied();
        match self {
            Self::SameType(a, b) => match (value(a), value(b)) {
                (Some(a), Some(b)) => graph.value(a).ty() == graph.value(b).ty(),
                _ => false,
            },
            Self::IdentityTranspose(capture) => {
                value(capture).is_some_and(|transposed| is_identity_transpose(graph, transposed))
            }
        }
    }
}

/// Whether `value` is the output of a transpose that keeps every axis.
fn is_identity_transpose(graph: &Graph, value: ValueId) -> bool {
    let node = graph.producer(value).and_then(|node| graph.node(node));
    node.is_some_and(|node| match node.op() {
        Op::Transpose(attrs) => {
            attrs.permutation.iter().enumerate().all(|(position, axis)| axis.index() == position)
        }
        _ => false,
    })
}
