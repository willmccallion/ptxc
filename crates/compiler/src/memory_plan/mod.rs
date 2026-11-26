//! Static memory planning from buffer lifetimes, as XLA's
//! `BufferAssignment`: each arena buffer is live from the first step touching
//! it to the last, and buffers live at the same step never share a byte.

mod greedy_by_size;
mod lifetime;
mod linear_scan;
mod planner;
mod verify;

pub use verify::ArenaError;

use crate::error::CompileError;
use greedy_by_size::GreedyBySize;
use lifetime::{lifetimes, peak_live_bytes};
use linear_scan::LinearScan;
use planner::{ArenaPlan, MemoryPlanner, Request};
use ptxc_core::{ByteSize, DType, ElementCount, IndexVec};
use ptxc_program::{Buffer, BufferId, Location, Program, Step};
use verify::{Placed, verify};

/// Where an emitted buffer is: fixed already, or in the arena wherever the
/// plan puts it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Place {
    /// At this location.
    Fixed(Location),
    /// In the arena.
    Arena,
}

/// A buffer before the arena is planned.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Unplaced {
    /// The element type.
    pub(crate) dtype: DType,
    /// The number of elements.
    pub(crate) elements: ElementCount,
    /// Where it is.
    pub(crate) place: Place,
}

/// The planners `place` runs, keeping the smallest arena; the first wins a
/// tie.
const PLANNERS: [&dyn MemoryPlanner; 2] = [&GreedyBySize, &LinearScan];

/// `buffers` located, each `Place::Arena` one in an arena planned so that
/// buffers `steps` touch at the same step never share a byte; and the
/// arena's size.
pub(crate) fn place(
    buffers: IndexVec<BufferId, Unplaced>,
    steps: &[Step],
) -> Result<(IndexVec<BufferId, Buffer>, ByteSize), CompileError> {
    let lifetimes = lifetimes(steps, &buffers);
    let mut requests = Vec::new();
    for (id, buffer) in buffers.iter_enumerated() {
        if buffer.place == Place::Arena {
            let size = buffer.elements.in_bytes(buffer.dtype)?;
            requests.push(Request { size, lifetime: lifetimes[id] });
        }
    }
    let plan = smallest_plan(&requests).ok_or(CompileError::TooLarge)?;
    let mut offsets = plan.offsets.into_iter();
    let located = buffers
        .into_iter()
        .map(|buffer| {
            let location = match buffer.place {
                Place::Fixed(location) => location,
                Place::Arena => Location::Arena(offsets.next().ok_or(CompileError::Unplaced)?),
            };
            Ok(Buffer { dtype: buffer.dtype, elements: buffer.elements, location })
        })
        .collect::<Result<_, CompileError>>()?;
    Ok((located, plan.size))
}

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

/// The plan for `requests` with the smallest arena, of those `PLANNERS`
/// make; `None` if none can place them below 2^64.
fn smallest_plan(requests: &[Request]) -> Option<ArenaPlan> {
    let plans = PLANNERS.iter().filter_map(|planner| planner.plan(requests));
    plans.min_by_key(|plan| plan.size)
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
