//! Safe bindings to the CUDA driver API, loaded at runtime with dlopen. The only crate with FFI.

mod context;
mod device;
mod error;
mod event;
mod graph;
mod launch;
mod memory;
mod module;
mod stream;
mod sys;
