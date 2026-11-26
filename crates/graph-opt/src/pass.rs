//! The `Pass` trait and `PassManager`, which verifies the graph after every
//! pass in debug builds and in tests.

use crate::passes::{Canonicalize, ConstFold, Cse, Dce};
use ptxc_core::SizeError;
use ptxc_graph::{CycleError, Graph, RewriteError, VerifyError};
use ptxc_interp::InterpError;
use thiserror::Error;

/// What a pass did to the graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PassReport {
    /// Whether the pass changed the graph.
    pub changed: bool,
}

/// Why a pass could not run.
#[derive(Debug, Error)]
pub enum PassError {
    /// A rewrite the graph refused.
    #[error(transparent)]
    Rewrite(#[from] RewriteError),
    /// The graph has a cycle.
    #[error(transparent)]
    Cycle(#[from] CycleError),
    /// The interpreter could not evaluate a node being folded.
    #[error(transparent)]
    Interpret(#[from] InterpError),
    /// A tensor's size overflows.
    #[error(transparent)]
    Size(#[from] SizeError),
}

/// A transformation of a whole graph, as MLIR's and XLA's passes.
pub trait Pass: std::fmt::Debug {
    /// The name the pass is reported by.
    fn name(&self) -> &'static str;

    /// Runs the pass over `graph`.
    fn run(&self, graph: &mut Graph) -> Result<PassReport, PassError>;
}

/// Why the pass manager stopped.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// A pass failed.
    #[error("{pass} failed: {source}")]
    Failed {
        /// The pass.
        pass: &'static str,
        /// Why.
        source: PassError,
    },
    /// A pass left a graph the verifier rejects.
    #[error("{pass} left an invalid graph: {source}")]
    Invalid {
        /// The pass.
        pass: &'static str,
        /// What the verifier found.
        source: VerifyError,
    },
}

/// When the pass manager runs the graph verifier after a pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verification {
    /// In debug builds.
    InDebugBuilds,
    /// Always, as a test asks.
    Always,
}

/// Runs passes in order, as MLIR's pass manager does, verifying the graph
/// after each one in debug builds or when asked.
#[derive(Debug)]
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    verification: Verification,
}

impl PassManager {
    /// A manager running `passes` in order.
    pub fn new(passes: Vec<Box<dyn Pass>>) -> Self {
        Self { passes, verification: Verification::InDebugBuilds }
    }

    /// The passes the compiler runs.
    pub fn standard() -> Self {
        Self::new(vec![Box::new(ConstFold), Box::new(Canonicalize), Box::new(Cse), Box::new(Dce)])
    }

    /// `self`, verifying the graph after every pass in release builds too.
    #[must_use]
    pub fn verifying_each_pass(self) -> Self {
        Self { passes: self.passes, verification: Verification::Always }
    }

    /// Runs each pass over `graph`, in order; each pass's report.
    pub fn run(&self, graph: &mut Graph) -> Result<Vec<PassReport>, PipelineError> {
        let verify = self.verification == Verification::Always || cfg!(debug_assertions);
        let mut reports = Vec::with_capacity(self.passes.len());
        for pass in &self.passes {
            let failed = |source| PipelineError::Failed { pass: pass.name(), source };
            reports.push(pass.run(graph).map_err(failed)?);
            if verify {
                let invalid = |source| PipelineError::Invalid { pass: pass.name(), source };
                graph.verify().map_err(invalid)?;
            }
        }
        Ok(reports)
    }
}
