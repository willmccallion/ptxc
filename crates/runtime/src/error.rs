//! `RuntimeError`: why a program did not load or run.

use ptxc_core::{BytesError, DType, Shape, TensorError};
use ptxc_cuda::{ComputeCapability, CudaError};
use ptxc_target::SmVersion;
use thiserror::Error;

/// Why a program did not load or run.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// A driver call failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),
    /// The program was compiled for another GPU.
    #[error("the program targets {program}, but the device is {}.{}; recompile for this GPU", .device.major, .device.minor)]
    Target {
        /// The program's target.
        program: SmVersion,
        /// The device's compute capability.
        device: ComputeCapability,
    },
    /// The wrong number of inputs.
    #[error("the program takes {expected} inputs, but {actual} were given")]
    InputCount {
        /// The program's inputs.
        expected: usize,
        /// The inputs given.
        actual: usize,
    },
    /// An input of the wrong type or shape.
    #[error("input `{name}` must be {dtype:?} of shape {shape:?}")]
    Input {
        /// The input's name.
        name: String,
        /// Its element type.
        dtype: DType,
        /// Its shape.
        shape: Shape,
    },
    /// A size beyond 64 bits.
    #[error("a buffer is too large for the device")]
    TooLarge,
    /// An output that does not fit its shape, which is a runtime bug.
    #[error(transparent)]
    Tensor(#[from] TensorError),
    /// Output bytes that do not decode, which is a runtime bug.
    #[error(transparent)]
    Bytes(#[from] BytesError),
}
