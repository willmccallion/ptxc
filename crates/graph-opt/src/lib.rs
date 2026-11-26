//! Graph optimisation: pass manager, declarative rewrite rules, constant folding, layout selection, fusion partitioning.

mod attention;
mod fusion;
mod pass;
mod passes;
mod rewrite;

pub use fusion::{GreedyPartitioner, OneNodePerGroup, Partitioner};
pub use pass::{Pass, PassError, PassManager, PassReport, PipelineError};
pub use passes::{Canonicalize, ConstFold, Cse, Dce, Simplify};
