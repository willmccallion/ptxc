//! `Buffer` and `MemScope` (global, shared, local).

use crate::compute::TensorName;
use ptxc_core::{DType, ElementCount};

ptxc_core::define_index!(
    /// A buffer of a lowered kernel.
    pub BufferId
);

/// Where a buffer lives.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MemScope {
    /// Device memory, passed as a kernel parameter.
    Global,
    /// Shared by a block's threads, allocated by an `Alloc`.
    Shared,
    /// Private to a thread, allocated by an `Alloc`.
    Local,
}

/// How a kernel uses a global buffer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Access {
    /// An input: only loaded.
    Read,
    /// An output: only stored to.
    Write,
}

/// A buffer: its name, element type, scope and size.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Buffer {
    /// The name it prints with.
    pub name: TensorName,
    /// The element type.
    pub dtype: DType,
    /// Where it lives.
    pub scope: MemScope,
    /// Its elements.
    pub size: ElementCount,
}
