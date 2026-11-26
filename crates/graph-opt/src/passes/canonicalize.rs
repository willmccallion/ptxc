//! Canonicalisation: each computation given one form.

use crate::pass::{Pass, PassError, PassReport};
use crate::rewrite::{GreedyEngine, RewriteEngine, canonical};
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
        let applied = GreedyEngine.apply(graph, &canonical())?;
        Ok(PassReport { changed: !applied.is_empty() })
    }
}
