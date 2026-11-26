//! Linear-scan offset assignment.

use super::lifetime::{Lifetime, StepIndex};
use super::planner::{ArenaPlan, MemoryPlanner, Request};
use ptxc_core::{ByteOffset, ByteSize};
use ptxc_program::BUFFER_ALIGNMENT;
use std::ops::Range;

/// Linear scan over the step order, as Poletto and Sarkar allocate
/// registers: buffers in the order they become live; before each, the bytes
/// of the buffers dead by its first step return to a free list, and it
/// takes the free range that fits it most tightly, or the arena grows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LinearScan;

impl MemoryPlanner for LinearScan {
    fn plan(&self, requests: &[Request]) -> Option<ArenaPlan> {
        let mut live: Vec<(usize, Lifetime)> = requests
            .iter()
            .enumerate()
            .filter_map(|(index, request)| Some((index, request.lifetime?)))
            .collect();
        live.sort_by_key(|&(index, lifetime)| (lifetime.first(), index));
        let mut offsets = vec![ByteOffset::new(0); requests.len()];
        let mut arena = Arena::default();
        for (index, lifetime) in live {
            arena.free_dead_before(lifetime.first());
            let bytes = arena.take(requests[index].size)?;
            offsets[index] = ByteOffset::new(bytes.start);
            arena.active.push((lifetime.last(), bytes));
        }
        let untouched = requests.iter().filter(|request| request.lifetime.is_none());
        let end = untouched.map(|request| request.size.get()).fold(arena.end, u64::max);
        Some(ArenaPlan { offsets, size: ByteSize::new(end) })
    }
}

/// The arena as the scan grows it.
#[derive(Default)]
struct Arena {
    /// Its end.
    end: u64,
    /// The bytes of each live buffer, with its last step.
    active: Vec<(StepIndex, Range<u64>)>,
    /// The ranges below the end no live buffer holds, sorted, none touching.
    free: Vec<Range<u64>>,
}

impl Arena {
    /// Frees the bytes of the live buffers whose last step is before `step`.
    fn free_dead_before(&mut self, step: StepIndex) {
        let (dead, live): (Vec<_>, Vec<_>) =
            self.active.drain(..).partition(|&(last, _)| last < step);
        self.active = live;
        for (_, bytes) in dead {
            self.release(bytes);
        }
    }

    /// Returns `bytes` to the free list, joined to the free ranges it
    /// touches.
    fn release(&mut self, bytes: Range<u64>) {
        let at = self.free.partition_point(|free| free.start < bytes.start);
        self.free.insert(at, bytes);
        if self.free.get(at + 1).is_some_and(|next| next.start == self.free[at].end) {
            let next = self.free.remove(at + 1);
            self.free[at].end = next.end;
        }
        if at > 0 && self.free[at - 1].end == self.free[at].start {
            let joined = self.free.remove(at);
            self.free[at - 1].end = joined.end;
        }
    }

    /// Aligned bytes for a buffer of `size`: in the smallest free range
    /// that fits it, or else from the end, which grows. `None` if the end
    /// passes 2^64.
    fn take(&mut self, size: ByteSize) -> Option<Range<u64>> {
        if let Some(index) = self.tightest_fit(size) {
            let free = self.free.remove(index);
            let start = aligned(free.start)?;
            self.release_unless_empty(free.start..start);
            self.release_unless_empty(start + size.get()..free.end);
            return Some(start..start + size.get());
        }
        let top = self.free.pop_if(|top| top.end == self.end);
        let below = top.map_or(self.end, |top| top.start);
        let start = aligned(below)?;
        self.release_unless_empty(below..start);
        self.end = start.checked_add(size.get())?;
        Some(start..self.end)
    }

    /// The free range, the smallest of those that fit, for a buffer of
    /// `size` at its first aligned offset.
    fn tightest_fit(&self, size: ByteSize) -> Option<usize> {
        let fits = |free: &Range<u64>| {
            aligned(free.start)
                .and_then(|start| start.checked_add(size.get()))
                .is_some_and(|end| end <= free.end)
        };
        let fitting = self.free.iter().enumerate().filter(|(_, free)| fits(free));
        fitting.min_by_key(|(_, free)| free.end - free.start).map(|(index, _)| index)
    }

    fn release_unless_empty(&mut self, bytes: Range<u64>) {
        if !bytes.is_empty() {
            self.release(bytes);
        }
    }
}

/// The first offset at or after `offset` aligned for any buffer.
fn aligned(offset: u64) -> Option<u64> {
    ByteOffset::new(offset).align_up(BUFFER_ALIGNMENT).ok().map(ByteOffset::get)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::LinearScan;
    use crate::memory_plan::lifetime::Lifetime;
    use crate::memory_plan::planner::{MemoryPlanner, Request};
    use ptxc_core::{ByteOffset, ByteSize};

    fn request(size: u64, steps: Option<(usize, usize)>) -> Request {
        Request {
            size: ByteSize::new(size),
            lifetime: steps.map(|(first, last)| Lifetime::between(first, last)),
        }
    }

    fn offsets(requests: &[Request]) -> Vec<u64> {
        LinearScan.plan(requests).unwrap().offsets.into_iter().map(ByteOffset::get).collect()
    }

    #[test]
    fn a_buffer_takes_the_bytes_of_one_dead_before_it() {
        let requests = [request(1000, Some((0, 1))), request(600, Some((2, 3)))];

        let plan = LinearScan.plan(&requests).unwrap();

        assert_eq!(offsets(&requests), [0, 0]);
        assert_eq!(plan.size, ByteSize::new(1000));
    }

    #[test]
    fn buffers_live_together_are_placed_in_the_order_they_become_live() {
        let requests = [request(100, Some((1, 2))), request(600, Some((0, 2)))];

        assert_eq!(offsets(&requests), [768, 0]);
    }

    #[test]
    fn a_buffer_takes_the_free_range_that_fits_it_most_tightly() {
        let requests = [
            request(1024, Some((0, 0))),
            request(256, Some((0, 9))),
            request(256, Some((0, 0))),
            request(256, Some((0, 9))),
            request(200, Some((1, 9))),
        ];

        assert_eq!(offsets(&requests), [0, 1024, 1280, 1536, 1280]);
    }

    #[test]
    fn neighbours_freed_together_join_into_one_range() {
        let requests = [
            request(256, Some((0, 0))),
            request(256, Some((0, 0))),
            request(256, Some((0, 9))),
            request(512, Some((1, 9))),
        ];

        assert_eq!(offsets(&requests), [0, 256, 512, 0]);
    }

    #[test]
    fn a_range_freed_before_a_free_neighbour_joins_it() {
        let requests = [
            request(256, Some((0, 1))),
            request(256, Some((0, 0))),
            request(256, Some((0, 9))),
            request(300, Some((1, 9))),
            request(512, Some((2, 9))),
        ];

        assert_eq!(offsets(&requests), [0, 256, 512, 768, 0]);
    }

    #[test]
    fn the_rest_of_a_range_a_buffer_takes_part_of_stays_free() {
        let requests = [
            request(1024, Some((0, 0))),
            request(256, Some((0, 9))),
            request(256, Some((1, 9))),
            request(512, Some((1, 9))),
        ];

        assert_eq!(offsets(&requests), [0, 1024, 0, 256]);
    }

    #[test]
    fn a_free_range_at_the_end_grows_into_the_buffer_that_needs_more() {
        let requests =
            [request(256, Some((0, 9))), request(256, Some((0, 0))), request(1000, Some((1, 9)))];

        let plan = LinearScan.plan(&requests).unwrap();

        assert_eq!(offsets(&requests), [0, 256, 256]);
        assert_eq!(plan.size, ByteSize::new(1256));
    }

    #[test]
    fn a_dead_empty_buffer_joins_the_free_range_it_starts() {
        let requests =
            [request(1000, Some((0, 1))), request(0, Some((2, 2))), request(2000, Some((3, 3)))];

        let plan = LinearScan.plan(&requests).unwrap();

        assert_eq!(offsets(&requests), [0, 0, 0]);
        assert_eq!(plan.size, ByteSize::new(2000));
    }

    #[test]
    fn a_buffer_no_step_touches_sits_at_the_start_inside_the_arena() {
        let requests = [request(300, Some((0, 0))), request(2000, None)];

        let plan = LinearScan.plan(&requests).unwrap();

        assert_eq!(offsets(&requests), [0, 0]);
        assert_eq!(plan.size, ByteSize::new(2000));
    }
}
