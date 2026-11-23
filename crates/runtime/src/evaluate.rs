//! `Evaluate` for the GPU runtime, so the differential tests compare it
//! with the interpreters through one interface.

use crate::error::RuntimeError;
use crate::executor::{Eager, Executor};
use ptxc_core::{Evaluate, HostTensor};

impl Evaluate for Executor<Eager> {
    type Error = RuntimeError;

    fn evaluate(&mut self, inputs: &[HostTensor]) -> Result<Vec<HostTensor>, RuntimeError> {
        self.run(inputs)
    }
}
