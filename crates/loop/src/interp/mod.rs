//! CPU interpreter for lowered kernels, so schedules are checked without a GPU.

mod grid;
mod memory;
mod race;
