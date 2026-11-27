//! Graph passes.

mod canonicalize;
mod const_fold;
mod cse;
mod dce;
mod layout;
mod simplify;

pub use canonicalize::Canonicalize;
pub use const_fold::ConstFold;
pub use cse::Cse;
pub use dce::Dce;
pub use simplify::Simplify;

use crate::pass::{PassError, PassReport};
use crate::rewrite::{GreedyEngine, RewriteEngine, RewriteRule};
use ptxc_graph::Graph;

/// Applies `rules` with the greedy engine, reporting whether any applied.
fn rewrite_greedily(graph: &mut Graph, rules: &[RewriteRule]) -> Result<PassReport, PassError> {
    let applied = GreedyEngine.apply(graph, rules)?;
    Ok(PassReport { changed: !applied.is_empty() })
}
