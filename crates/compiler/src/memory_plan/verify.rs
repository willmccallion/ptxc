//! The independent check of an arena plan, as XLA checks its heap
//! simulator's: buffers live at the same step never share a byte, and every
//! buffer is aligned and inside the arena.

use super::lifetime::{Lifetime, live_together};
use ptxc_core::{ByteOffset, ByteSize, SizeError};
use ptxc_program::{BUFFER_ALIGNMENT, BufferId};
use std::ops::Range;
use thiserror::Error;

/// A buffer placed in the arena: where, how large, and when it is live.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Placed {
    /// The buffer.
    pub(crate) buffer: BufferId,
    /// Its offset in the arena.
    pub(crate) offset: ByteOffset,
    /// Its size.
    pub(crate) size: ByteSize,
    /// When it is live; `None` if no step touches it.
    pub(crate) lifetime: Option<Lifetime>,
}

/// Why an arena plan is wrong.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Error)]
pub enum ArenaError {
    /// Two buffers live at the same step share bytes.
    #[error("{first:?} and {second:?} are live at the same step and share bytes")]
    Overlap {
        /// The buffer placed first in the plan.
        first: BufferId,
        /// The buffer placed later.
        second: BufferId,
    },
    /// A buffer's offset is not aligned.
    #[error("{0:?} is at {1:?}, which is not aligned to 256 bytes")]
    Unaligned(BufferId, ByteOffset),
    /// A buffer runs past the end of the arena.
    #[error("{0:?} runs past the end of the arena")]
    OutOfBounds(BufferId),
    /// A buffer's size overflows.
    #[error(transparent)]
    Size(#[from] SizeError),
    /// The bytes live at one step pass 2^64.
    #[error("the bytes live at one step pass 2^64")]
    TooLarge,
}

/// Whether `placed` fit an arena of `arena` bytes with no two live buffers
/// sharing a byte.
pub(crate) fn verify(placed: &[Placed], arena: ByteSize) -> Result<(), ArenaError> {
    let bytes =
        placed.iter().map(|buffer| bytes_in(buffer, arena)).collect::<Result<Vec<_>, _>>()?;
    for (index, (first, first_bytes)) in placed.iter().zip(&bytes).enumerate() {
        for (second, second_bytes) in placed.iter().zip(&bytes).skip(index + 1) {
            if live_together(first.lifetime, second.lifetime) && share(first_bytes, second_bytes) {
                return Err(ArenaError::Overlap { first: first.buffer, second: second.buffer });
            }
        }
    }
    Ok(())
}

/// The bytes `buffer` takes, if it is aligned and inside the arena.
const fn bytes_in(buffer: &Placed, arena: ByteSize) -> Result<Range<u64>, ArenaError> {
    if !buffer.offset.is_aligned(BUFFER_ALIGNMENT) {
        return Err(ArenaError::Unaligned(buffer.buffer, buffer.offset));
    }
    let start = buffer.offset.get();
    match start.checked_add(buffer.size.get()) {
        Some(end) if end <= arena.get() => Ok(start..end),
        _ => Err(ArenaError::OutOfBounds(buffer.buffer)),
    }
}

/// Whether two ranges have a byte in common; an empty range has none.
fn share(first: &Range<u64>, second: &Range<u64>) -> bool {
    first.start.max(second.start) < first.end.min(second.end)
}

#[cfg(test)]
mod tests {
    use super::{ArenaError, Placed, verify};
    use crate::memory_plan::lifetime::Lifetime;
    use ptxc_core::{ByteOffset, ByteSize, Idx};
    use ptxc_program::BufferId;

    fn placed(buffer: usize, offset: u64, size: u64, steps: Option<(usize, usize)>) -> Placed {
        Placed {
            buffer: BufferId::new(buffer),
            offset: ByteOffset::new(offset),
            size: ByteSize::new(size),
            lifetime: steps.map(|(first, last)| Lifetime::between(first, last)),
        }
    }

    #[test]
    fn buffers_that_end_where_the_next_begins_share_no_byte() {
        let plan = [placed(0, 0, 256, Some((0, 3))), placed(1, 256, 100, Some((0, 3)))];

        assert_eq!(verify(&plan, ByteSize::new(356)), Ok(()));
    }

    #[test]
    fn a_buffer_that_ends_where_an_earlier_one_begins_shares_no_byte() {
        let plan = [placed(0, 256, 100, Some((0, 3))), placed(1, 0, 256, Some((0, 3)))];

        assert_eq!(verify(&plan, ByteSize::new(356)), Ok(()));
    }

    #[test]
    fn a_buffer_reaching_into_a_live_neighbour_overlaps_it() {
        let plan = [placed(0, 0, 257, Some((0, 3))), placed(1, 256, 100, Some((3, 4)))];

        let found = verify(&plan, ByteSize::new(356));

        assert_eq!(
            found,
            Err(ArenaError::Overlap { first: BufferId::new(0), second: BufferId::new(1) })
        );
    }

    #[test]
    fn an_empty_buffer_inside_a_live_one_shares_no_byte() {
        let plan = [placed(0, 0, 2049, Some((35, 37))), placed(1, 256, 0, Some((33, 35)))];

        assert_eq!(verify(&plan, ByteSize::new(2049)), Ok(()));
    }

    #[test]
    fn a_buffer_no_step_touches_overlaps_nothing() {
        let plan = [placed(0, 0, 100, Some((0, 3))), placed(1, 0, 100, None)];

        assert_eq!(verify(&plan, ByteSize::new(100)), Ok(()));
    }

    #[test]
    fn an_offset_off_the_alignment_is_refused() {
        let plan = [placed(0, 128, 16, Some((0, 0)))];

        let found = verify(&plan, ByteSize::new(256));

        assert_eq!(found, Err(ArenaError::Unaligned(BufferId::new(0), ByteOffset::new(128))));
    }

    #[test]
    fn a_buffer_past_the_arena_is_refused() {
        let plan = [placed(0, 256, 16, Some((0, 0)))];

        assert_eq!(
            verify(&plan, ByteSize::new(271)),
            Err(ArenaError::OutOfBounds(BufferId::new(0)))
        );
        assert_eq!(verify(&plan, ByteSize::new(272)), Ok(()));
    }

    #[test]
    fn a_buffer_whose_end_passes_2_to_the_64_is_refused() {
        let plan = [placed(0, 256, u64::MAX, Some((0, 0)))];

        assert_eq!(
            verify(&plan, ByteSize::new(u64::MAX)),
            Err(ArenaError::OutOfBounds(BufferId::new(0)))
        );
    }
}
