//! Safe bindings to the CUDA driver API, loaded at runtime with dlopen. The only crate with FFI.
//!
//! Launching a kernel is safe although Rust cannot check what the kernel
//! does: this crate never maps host memory into the device's address space
//! (no pinned, mapped or managed host memory), so a wrong kernel can corrupt
//! only device memory, and fails with a CUDA error rather than undefined
//! behaviour on the host. Adding host memory the device can reach would
//! break this, and needs a new decision.

#![allow(
    unsafe_code,
    reason = "the driver API is reached through FFI (D8); ARCHITECTURE.md rule 4"
)]

mod context;
mod device;
mod driver;
mod error;
mod event;
mod graph;
mod launch;
mod memory;
mod module;
mod stream;
mod sys;

pub use context::Context;
pub use device::{ComputeCapability, Device, DeviceOrdinal};
pub use driver::Driver;
pub use error::CudaError;
pub use event::Event;
pub use graph::{CapturedGraph, GraphCapture, GraphExec};
pub use launch::{BlockDim, GridDim, KernelArg, Launch, Unset};
pub use memory::{DeviceAddress, DeviceBuffer, DeviceSlice};
pub use module::{Function, Module};
pub use stream::Stream;
pub use sys::Library;
