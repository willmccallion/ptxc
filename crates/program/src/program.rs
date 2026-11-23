//! `Program`: everything the runtime needs, with no reference to any IR,
//! checked once when it is made.

use crate::buffer_plan::{BUFFER_ALIGNMENT, Buffer, BufferId, InputIndex, Location, OutputIndex};
use crate::kernel::{KernelId, KernelSpec};
use crate::step::Step;
use ptxc_core::{ByteOffset, ByteSize, IndexVec, Shape, SizeError};
use ptxc_target::SmVersion;
use thiserror::Error;

/// A program input or output: its name, its shape, and the buffer holding
/// it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IoBinding {
    /// The graph's name for it.
    pub name: String,
    /// Its shape; the buffer has its element type.
    pub shape: Shape,
    /// The buffer it binds.
    pub buffer: BufferId,
}

/// A program's pieces, before they are checked against each other.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ProgramParts {
    /// The compute capability its kernels target.
    pub target: SmVersion,
    /// Its kernels.
    pub kernels: IndexVec<KernelId, KernelSpec>,
    /// The constant blob, uploaded once at load.
    pub constants: Vec<u8>,
    /// The size of the arena of intermediates.
    pub arena: ByteSize,
    /// Every buffer the steps touch.
    pub buffers: IndexVec<BufferId, Buffer>,
    /// The steps, in order.
    pub steps: Vec<Step>,
    /// The inputs, in order.
    pub inputs: IndexVec<InputIndex, IoBinding>,
    /// The outputs, in order.
    pub outputs: IndexVec<OutputIndex, IoBinding>,
}

/// Pieces that do not form a program.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Error)]
pub enum ProgramError {
    /// A step names a kernel the program does not have.
    #[error("a step launches {0:?}, which does not exist")]
    Kernel(KernelId),
    /// A step or binding names a buffer the program does not have.
    #[error("{0:?} does not exist")]
    Buffer(BufferId),
    /// A buffer's size overflows.
    #[error("{0:?}'s size overflows: {1}")]
    Size(BufferId, SizeError),
    /// An arena or constant offset is not aligned.
    #[error("{0:?} is at {1:?}, which is not aligned to 256 bytes")]
    Unaligned(BufferId, ByteOffset),
    /// A buffer runs past the end of the arena or the constant blob.
    #[error("{0:?} runs past the end of its arena or constant blob")]
    OutOfBounds(BufferId),
    /// An input or output's buffer is not at its own position.
    #[error("{0:?} is bound to an input or output it is not located at")]
    Binding(BufferId),
    /// An input or output's shape has a different element count from its
    /// buffer.
    #[error("{0:?}'s binding has a different element count")]
    Shape(BufferId),
    /// A copy between buffers of different sizes.
    #[error("a copy from {from:?} to {to:?} changes size")]
    CopySize {
        /// The source.
        from: BufferId,
        /// The destination.
        to: BufferId,
    },
}

/// A compiled model the runtime can load and run, every reference in it
/// checked.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Program {
    parts: ProgramParts,
}

impl Program {
    /// Checks `parts` against each other.
    pub fn new(parts: ProgramParts) -> Result<Self, ProgramError> {
        for (id, buffer) in parts.buffers.iter_enumerated() {
            check_location(&parts, id, *buffer)?;
        }
        for (index, binding) in parts.inputs.iter_enumerated() {
            check_binding(&parts, binding, Location::Input(index))?;
        }
        for (index, binding) in parts.outputs.iter_enumerated() {
            check_binding(&parts, binding, Location::Output(index))?;
        }
        for step in &parts.steps {
            check_step(&parts, step)?;
        }
        Ok(Self { parts })
    }

    /// The compute capability its kernels target.
    pub const fn target(&self) -> SmVersion {
        self.parts.target
    }

    /// Its kernels.
    pub const fn kernels(&self) -> &IndexVec<KernelId, KernelSpec> {
        &self.parts.kernels
    }

    /// The constant blob.
    pub fn constants(&self) -> &[u8] {
        &self.parts.constants
    }

    /// The size of the arena of intermediates.
    pub const fn arena(&self) -> ByteSize {
        self.parts.arena
    }

    /// Every buffer the steps touch.
    pub const fn buffers(&self) -> &IndexVec<BufferId, Buffer> {
        &self.parts.buffers
    }

    /// The steps, in order.
    pub fn steps(&self) -> &[Step] {
        &self.parts.steps
    }

    /// The inputs, in order.
    pub const fn inputs(&self) -> &IndexVec<InputIndex, IoBinding> {
        &self.parts.inputs
    }

    /// The outputs, in order.
    pub const fn outputs(&self) -> &IndexVec<OutputIndex, IoBinding> {
        &self.parts.outputs
    }
}

fn check_location(parts: &ProgramParts, id: BufferId, buffer: Buffer) -> Result<(), ProgramError> {
    let bytes = buffer.elements.in_bytes(buffer.dtype).map_err(|e| ProgramError::Size(id, e))?;
    let offset = match buffer.location {
        Location::Arena(offset) | Location::Constant(offset) => offset,
        Location::Input(index) => return located(parts.inputs.get(index), id),
        Location::Output(index) => return located(parts.outputs.get(index), id),
    };
    if !offset.is_aligned(BUFFER_ALIGNMENT) {
        return Err(ProgramError::Unaligned(id, offset));
    }
    let end = offset.get().checked_add(bytes.get()).ok_or(ProgramError::OutOfBounds(id))?;
    let fits = match buffer.location {
        Location::Constant(_) => usize::try_from(end).is_ok_and(|end| end <= parts.constants.len()),
        _ => end <= parts.arena.get(),
    };
    if fits { Ok(()) } else { Err(ProgramError::OutOfBounds(id)) }
}

/// Whether `binding`, the input or output a buffer is located at, binds
/// that buffer.
fn located(binding: Option<&IoBinding>, id: BufferId) -> Result<(), ProgramError> {
    match binding {
        Some(binding) if binding.buffer == id => Ok(()),
        _ => Err(ProgramError::Binding(id)),
    }
}

fn check_binding(
    parts: &ProgramParts,
    binding: &IoBinding,
    at: Location,
) -> Result<(), ProgramError> {
    let buffer = buffer(parts, binding.buffer)?;
    if buffer.location != at {
        return Err(ProgramError::Binding(binding.buffer));
    }
    if binding.shape.element_count() != buffer.elements {
        return Err(ProgramError::Shape(binding.buffer));
    }
    Ok(())
}

fn check_step(parts: &ProgramParts, step: &Step) -> Result<(), ProgramError> {
    match step {
        Step::Launch { kernel, args } => {
            if parts.kernels.get(*kernel).is_none() {
                return Err(ProgramError::Kernel(*kernel));
            }
            args.iter().try_for_each(|&arg| buffer(parts, arg).map(|_| ()))
        }
        Step::Copy { from, to } => {
            let (source, destination) = (buffer(parts, *from)?, buffer(parts, *to)?);
            let size =
                |b: Buffer, id| b.elements.in_bytes(b.dtype).map_err(|e| ProgramError::Size(id, e));
            if size(source, *from)? == size(destination, *to)? {
                Ok(())
            } else {
                Err(ProgramError::CopySize { from: *from, to: *to })
            }
        }
    }
}

fn buffer(parts: &ProgramParts, id: BufferId) -> Result<Buffer, ProgramError> {
    parts.buffers.get(id).copied().ok_or(ProgramError::Buffer(id))
}
