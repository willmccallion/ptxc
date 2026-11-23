//! `Step`: one launch or copy, in execution order.

use crate::buffer_plan::BufferId;
use crate::kernel::KernelId;

/// One thing a program does, in order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Step {
    /// Launches `kernel` with these buffers as its pointer arguments.
    Launch {
        /// The kernel.
        kernel: KernelId,
        /// Its arguments, in its parameter order.
        args: Vec<BufferId>,
    },
    /// Copies one buffer's bytes into another of the same size, as when an
    /// output is a reshaped input.
    Copy {
        /// The source.
        from: BufferId,
        /// The destination.
        to: BufferId,
    },
}
