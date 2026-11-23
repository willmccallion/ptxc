//! `KernelSpec`: a kernel's PTX source, entry name and launch shape.

use ptxc_core::{KernelName, define_index};
use ptxc_target::{BlockCount, ThreadsPerBlock};

define_index!(
    /// A kernel in a program.
    pub KernelId
);

/// The grid and block a kernel launches with, x then y then z.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LaunchConfig {
    /// Blocks along each axis.
    pub grid: [BlockCount; 3],
    /// Threads along each axis.
    pub block: [ThreadsPerBlock; 3],
}

/// A kernel: the PTX module defining it, its entry point, and how it
/// launches.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct KernelSpec {
    /// The entry point's name.
    pub name: KernelName,
    /// The PTX module's text.
    pub ptx: String,
    /// The launch shape.
    pub launch: LaunchConfig,
}
