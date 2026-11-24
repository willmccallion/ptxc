//! `Buffer` and `MemScope` (global, shared, local, fragment).

use crate::compute::TensorName;
use ptxc_core::{DType, Dim, ElementCount};

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
    /// A logical tile one warp holds, its elements spread over the warp's
    /// registers in a layout only code generation knows, allocated by an
    /// `Alloc` and touched only by the fragment statements.
    Fragment(FragmentShape),
}

/// The rows and columns of a fragment's logical tile, row-major.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FragmentShape {
    /// Rows.
    pub rows: Dim,
    /// Columns.
    pub columns: Dim,
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
