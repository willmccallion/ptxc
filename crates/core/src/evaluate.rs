//! The `Evaluate` trait every executor implements, so differential tests are written once.

use crate::HostTensor;

/// Computes a fixed program's outputs from its inputs.
///
/// Each executor is built from what it runs: the reference interpreter from a
/// graph, the GPU executor from a compiled program, a cache from recorded
/// outputs.
pub trait Evaluate {
    /// Why an evaluation failed.
    type Error: std::error::Error;

    /// The outputs for `inputs`, both in the program's declared order.
    fn evaluate(&mut self, inputs: &[HostTensor]) -> Result<Vec<HostTensor>, Self::Error>;
}
