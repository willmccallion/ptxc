//! `CudaError`: the driver error name, the call that failed and its context.

use ptxc_core::{ByteOffset, ByteSize, KernelName};
use thiserror::Error;

/// Why a CUDA operation failed.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CudaError {
    /// `libcuda.so.1` could not be loaded, or lacks a function ptxc calls.
    #[error("cannot load the CUDA driver: {0}")]
    Load(String),
    /// A driver call returned an error.
    #[error("{call} failed: {name}{}", .detail.as_deref().map(|detail| format!("\n{detail}")).unwrap_or_default())]
    Driver {
        /// The driver function.
        call: &'static str,
        /// The `CUresult` value.
        code: i32,
        /// Its name, such as `CUDA_ERROR_OUT_OF_MEMORY`.
        name: String,
        /// What the driver said beyond the code, such as the JIT's log.
        detail: Option<String>,
    },
    /// A device attribute outside the range the driver documents.
    #[error("device attribute {attribute} is {value}, out of range")]
    Attribute {
        /// The attribute's number.
        attribute: i32,
        /// Its value.
        value: i32,
    },
    /// A slice that runs past its buffer's end.
    #[error("bytes {} to {} are past the end of a {}-byte buffer", .offset.get(), .offset.get().saturating_add(.size.get()), .buffer.get())]
    SliceBounds {
        /// The slice's start.
        offset: ByteOffset,
        /// The slice's length.
        size: ByteSize,
        /// The buffer's length.
        buffer: ByteSize,
    },
    /// A device-to-device copy between slices of different lengths.
    #[error("cannot copy {} bytes into {}", .from.get(), .to.get())]
    SliceSizes {
        /// The source's length.
        from: ByteSize,
        /// The destination's length.
        to: ByteSize,
    },
    /// A kernel launched with the wrong number of arguments.
    #[error("kernel {kernel} takes {expected} arguments, but {actual} were given")]
    ArgumentCount {
        /// The kernel.
        kernel: KernelName,
        /// Its parameters.
        expected: usize,
        /// The arguments given.
        actual: usize,
    },
    /// A kernel argument whose size is not its parameter's.
    #[error(
        "kernel {kernel} argument {index} is {} bytes, but its parameter is {}",
        .actual.get(),
        .expected.get()
    )]
    ArgumentSize {
        /// The kernel.
        kernel: KernelName,
        /// The argument's position.
        index: usize,
        /// The parameter's size.
        expected: ByteSize,
        /// The argument's size.
        actual: ByteSize,
    },
    /// A buffer, stream or event used with a context it does not belong to.
    #[error("{0} belongs to a different context")]
    ForeignContext(&'static str),
    /// A host-device copy whose sides differ in size.
    #[error("copying {} host bytes to or from a device buffer of {}", .host.get(), .device.get())]
    CopySize {
        /// The host slice's size.
        host: ByteSize,
        /// The device buffer's size.
        device: ByteSize,
    },
    /// A size the driver API cannot take.
    #[error("{} bytes is too large for the driver API", .0.get())]
    TooLarge(ByteSize),
    /// A string passed to the driver contains a NUL byte.
    #[error("the {0} contains a NUL byte")]
    InteriorNul(&'static str),
}
