//! Static memory planning from buffer lifetimes, as XLA's
//! `BufferAssignment`: each arena buffer is live from the first step touching
//! it to the last, and buffers live at the same step never share a byte.

mod greedy_by_size;
mod lifetime;
mod linear_scan;
mod planner;
mod verify;

pub use verify::ArenaError;

use lifetime::{lifetimes, peak_live_bytes};
use ptxc_core::ByteSize;
use ptxc_program::{Location, Program};
use verify::{Placed, verify};

/// A program's arena against the least any placement of its buffers needs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArenaUse {
    /// The arena's size.
    pub size: ByteSize,
    /// The most bytes the arena's buffers hold live at any one step.
    pub lower_bound: ByteSize,
}

/// Checks `program`'s arena: no two buffers live at the same step share a
/// byte, and each is aligned and inside the arena.
pub fn verify_arena(program: &Program) -> Result<(), ArenaError> {
    verify(&placed(program)?, program.arena())
}

/// `program`'s arena and its lower bound.
pub fn arena_use(program: &Program) -> Result<ArenaUse, ArenaError> {
    let live =
        placed(program)?.into_iter().filter_map(|buffer| Some((buffer.size, buffer.lifetime?)));
    let lower_bound = peak_live_bytes(live).ok_or(ArenaError::TooLarge)?;
    Ok(ArenaUse { size: program.arena(), lower_bound })
}

/// `program`'s arena buffers, with their lifetimes over its steps.
fn placed(program: &Program) -> Result<Vec<Placed>, ArenaError> {
    let lifetimes = lifetimes(program.steps(), program.buffers());
    let mut placed = Vec::new();
    for (buffer, found) in program.buffers().iter_enumerated() {
        let Location::Arena(offset) = found.location else { continue };
        let size = found.elements.in_bytes(found.dtype)?;
        placed.push(Placed { buffer, offset, size, lifetime: lifetimes[buffer] });
    }
    Ok(placed)
}
