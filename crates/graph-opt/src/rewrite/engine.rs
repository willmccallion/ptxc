//! The `RewriteEngine` trait, implemented by the greedy engine and later an e-graph engine.

use super::rule::RewriteRule;
use crate::pass::PassError;
use ptxc_graph::Graph;

/// Applies rewrite rules to a graph until none matches.
pub(crate) trait RewriteEngine {
    /// Rewrites `graph` with `rules`; the names of the rules applied, in
    /// order.
    fn apply(
        &self,
        graph: &mut Graph,
        rules: &[RewriteRule],
    ) -> Result<Vec<&'static str>, PassError>;
}
