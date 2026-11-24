//! CPU interpreter for lowered kernels, so schedules are checked without a GPU.
//!
//! Each block runs its threads in lockstep, statement by statement, with
//! the threads an `If` admits active; every value is rounded or wrapped to
//! its type at each operation, as a GPU computes it. Memory starts unset,
//! so reading an element no one stored, or finishing with an output element
//! unwritten, is an error; two threads touching one shared element between
//! barriers, one writing, or writing one global element, race.

mod grid;
mod memory;
mod ops;
mod race;
mod tensors;
mod value;
mod warp;

use crate::compute::TensorName;
use crate::stmt::{Access, BufferId, LoweredKernel};
use crate::verify::{VerifyError, verify};
use grid::Grid;
use memory::Memory;
use ptxc_core::{DType, HostTensor};
use std::collections::HashMap;
use thiserror::Error;

/// Why a kernel could not be run to completion.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum RunError {
    /// The kernel breaks a rule the verifier checks.
    #[error("the kernel does not verify: {0}")]
    Invalid(#[from] VerifyError),
    /// Inputs that do not match the kernel's input parameters.
    #[error("input {index} should be {dtype:?} with {elements} elements")]
    Input {
        /// The input's position.
        index: usize,
        /// The element type expected.
        dtype: DType,
        /// The elements expected.
        elements: u64,
    },
    /// The wrong number of inputs.
    #[error("the kernel takes {expected} inputs, given {actual}")]
    InputCount {
        /// The inputs it takes.
        expected: usize,
        /// The inputs given.
        actual: usize,
    },
    /// An element read before anything stored it.
    #[error("{buffer}[{index}] is read before it is stored")]
    Unset {
        /// The buffer.
        buffer: TensorName,
        /// The element.
        index: i64,
    },
    /// An output element no thread stored.
    #[error("output {buffer}[{index}] is never stored")]
    Unwritten {
        /// The output.
        buffer: TensorName,
        /// The element.
        index: i64,
    },
    /// An access outside its buffer.
    #[error("{buffer}[{index}] is outside the buffer")]
    OutOfBounds {
        /// The buffer.
        buffer: TensorName,
        /// The element.
        index: i64,
    },
    /// Two threads touching one element without a barrier between, one of
    /// them writing; or two threads writing one global element.
    #[error("threads race on {buffer}[{index}]")]
    Race {
        /// The buffer.
        buffer: TensorName,
        /// The element.
        index: i64,
    },
    /// A fragment statement some lanes of a warp did not reach, or a warp
    /// the block's threads do not fill.
    #[error("a fragment statement is reached by only some lanes of a warp")]
    PartialWarp,
    /// Lanes of one warp that put a fragment's tile at different places.
    #[error("the lanes of a warp disagree on where a fragment's tile is")]
    WarpDisagrees,
    /// A barrier some threads of the block did not reach.
    #[error("a barrier is reached by only some threads of a block")]
    DivergentBarrier,
    /// A buffer used outside its allocation.
    #[error("{0} is used where it is not allocated")]
    NotAllocated(TensorName),
    /// An index that overflows or divides by zero.
    #[error("an index overflows or divides by zero")]
    Index,
    /// An integer division or remainder by zero.
    #[error("integer division by zero")]
    DivisionByZero,
}

/// Runs `kernel` on `inputs`, one per input parameter in order, and returns
/// each output as a flat tensor of its buffer's size.
pub fn run(kernel: &LoweredKernel, inputs: &[HostTensor]) -> Result<Vec<HostTensor>, RunError> {
    verify(kernel)?;
    execute(kernel, inputs)
}

/// `kernel`'s outputs for `inputs`, the kernel trusted to verify.
fn execute(kernel: &LoweredKernel, inputs: &[HostTensor]) -> Result<Vec<HostTensor>, RunError> {
    let mut globals = HashMap::new();
    let input_params: Vec<_> =
        kernel.params().iter().filter(|param| param.access == Access::Read).collect();
    if input_params.len() != inputs.len() {
        return Err(RunError::InputCount { expected: input_params.len(), actual: inputs.len() });
    }
    for (index, (param, input)) in input_params.iter().zip(inputs).enumerate() {
        let buffer = &kernel.buffers()[param.buffer];
        if input.dtype() != buffer.dtype || input.shape().element_count() != buffer.size {
            return Err(RunError::Input {
                index,
                dtype: buffer.dtype,
                elements: buffer.size.get(),
            });
        }
        let _new = globals
            .insert(param.buffer, Memory::of(buffer.name.clone(), tensors::values(input.data())));
    }
    for param in kernel.params().iter().filter(|param| param.access == Access::Write) {
        let buffer = &kernel.buffers()[param.buffer];
        let size = usize::try_from(buffer.size.get()).unwrap_or(usize::MAX);
        let _new = globals.insert(param.buffer, Memory::unset(buffer.name.clone(), size));
    }
    let mut grid = Grid { kernel, globals, writers: race::GlobalWriters::default() };
    let launch = kernel.launch();
    let blocks: u64 = launch.grid.iter().map(|count| u64::from(count.get())).product();
    for block in 0..blocks {
        grid.run_block(usize::try_from(block).unwrap_or(usize::MAX))?;
    }
    outputs(kernel, &grid.globals)
}

fn outputs(
    kernel: &LoweredKernel,
    globals: &HashMap<BufferId, Memory>,
) -> Result<Vec<HostTensor>, RunError> {
    let mut outputs = Vec::new();
    for param in kernel.params().iter().filter(|param| param.access == Access::Write) {
        let buffer = &kernel.buffers()[param.buffer];
        let Some(memory) = globals.get(&param.buffer) else { continue };
        let values = memory.values()?;
        let tensor = HostTensor::vector(tensors::data(&values, buffer.dtype));
        outputs.push(tensor);
    }
    Ok(outputs)
}

#[cfg(test)]
mod fragment_tests;
#[cfg(test)]
mod semantics;
#[cfg(test)]
mod tests;
