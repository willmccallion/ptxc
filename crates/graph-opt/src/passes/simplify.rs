//! Algebraic simplification, batch normalisations folded into convolutions.

use crate::pass::{Pass, PassError, PassReport};
use crate::rewrite::{GreedyEngine, RewriteEngine, algebraic};
use ptxc_graph::Graph;

/// Applies the algebraic rewrite rules until none matches, as XLA's
/// algebraic simplifier does; constant folding then computes what they
/// leave, such as folded weights.
#[derive(Clone, Copy, Debug)]
pub struct Simplify;

impl Pass for Simplify {
    fn name(&self) -> &'static str {
        "simplify"
    }

    fn run(&self, graph: &mut Graph) -> Result<PassReport, PassError> {
        let applied = GreedyEngine.apply(graph, &algebraic())?;
        Ok(PassReport { changed: !applied.is_empty() })
    }
}
