//! `Buffer`: every tensor a program's steps touch, and where it lives: in
//! the arena of intermediates, in the constant blob, or in memory the
//! caller gives for an input or output.

use ptxc_core::{Alignment, ByteOffset, DType, ElementCount, define_index};

define_index!(
    /// A buffer in a program.
    pub BufferId
);

define_index!(
    /// A program input, by position.
    pub InputIndex
);

define_index!(
    /// A program output, by position.
    pub OutputIndex
);

/// The alignment of every arena and constant offset: `cudaMalloc`'s, so a
/// buffer at an offset is as aligned as a buffer of its own.
pub const BUFFER_ALIGNMENT: Alignment = Alignment::CUDA_MALLOC;

/// Where a buffer's bytes are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Location {
    /// At this offset in the arena.
    Arena(ByteOffset),
    /// At this offset in the constant blob, uploaded once at load.
    Constant(ByteOffset),
    /// The caller's memory for this input.
    Input(InputIndex),
    /// The caller's memory for this output.
    Output(OutputIndex),
}

/// A tensor's storage: its element type, its length and where it is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Buffer {
    /// The element type.
    pub dtype: DType,
    /// The number of elements.
    pub elements: ElementCount,
    /// Where its bytes are.
    pub location: Location,
}
