//! `Template`: the right-hand side of a rule, instantiated from a match.

use super::pattern::{Bindings, Capture};
use ptxc_graph::ValueId;

/// What a match is replaced with.
// NOTE: MLIR's declarative rewrites build new ops in their result
// patterns too; the canonical rules only forward a value, and building
// nodes comes with the rules that need it (#200, #201).
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Template {
    /// A value the pattern bound.
    Captured(Capture),
}

impl Template {
    /// The value replacing the match, if `bindings` has every value the
    /// template reads.
    pub(crate) fn instantiate(&self, bindings: &Bindings) -> Option<ValueId> {
        match self {
            Self::Captured(capture) => bindings.get(capture).copied(),
        }
    }
}
