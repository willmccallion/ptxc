//! `CompileError`: why a model did not compile, from whichever stage
//! refused it.

use ptxc_codegen::CodegenError;
use ptxc_core::SizeError;
use ptxc_graph::{CycleError, ValueId};
use ptxc_loop::ScheduleError;
use ptxc_lower::LowerError;
use ptxc_onnx::ImportError;
use ptxc_program::ProgramError;
use thiserror::Error;

/// Why a model did not compile.
#[derive(Debug, Error)]
pub enum CompileError {
    /// The ONNX model did not import.
    #[error(transparent)]
    Import(#[from] ImportError),
    /// The graph has a cycle.
    #[error(transparent)]
    Partition(#[from] CycleError),
    /// A group did not lower.
    #[error(transparent)]
    Lower(#[from] LowerError),
    /// A stage's default schedule did not apply.
    #[error(transparent)]
    Schedule(#[from] ScheduleError),
    /// A kernel did not generate.
    #[error(transparent)]
    Codegen(#[from] CodegenError),
    /// A buffer's size overflows.
    #[error(transparent)]
    Size(#[from] SizeError),
    /// The buffers do not fit 64-bit byte offsets.
    #[error("the model's buffers are too large to place")]
    TooLarge,
    /// A stage's kernel has parameters its stage binds no tensor to, which
    /// is a compiler bug.
    #[error("a kernel's parameters do not match its stage's tensors")]
    StageArity,
    /// A graph input or output has no name to bind it by.
    #[error("graph input or output {0:?} has no name")]
    Unnamed(ValueId),
    /// The emitted pieces do not form a program, which is a compiler bug.
    #[error("the compiler emitted an inconsistent program: {0}")]
    Program(#[from] ProgramError),
}
