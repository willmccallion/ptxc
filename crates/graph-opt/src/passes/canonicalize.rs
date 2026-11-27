//! Canonicalisation: each computation given one form.

use super::rewrite_greedily;
use crate::pass::{Pass, PassError, PassReport};
use crate::rewrite::canonical;
use ptxc_graph::Graph;

/// Applies the canonical rewrite rules until none matches, as MLIR's
/// canonicalizer pass does.
#[derive(Clone, Copy, Debug)]
pub struct Canonicalize;

impl Pass for Canonicalize {
    fn name(&self) -> &'static str {
        "canonicalize"
    }

    fn run(&self, graph: &mut Graph) -> Result<PassReport, PassError> {
        rewrite_greedily(graph, &canonical())
    }
}
