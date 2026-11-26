//! Greedy-by-size offset assignment.

use super::lifetime::live_together;
use super::planner::{ArenaPlan, MemoryPlanner, Request};
use ptxc_core::{ByteOffset, ByteSize};
use ptxc_program::BUFFER_ALIGNMENT;
use std::cmp::Reverse;
use std::ops::Range;

/// TensorFlow Lite's greedy-by-size planner (Pisarchyk and Lee, "Efficient
/// Memory Management for Deep Neural Net Inference"): buffers from largest
/// to smallest, each in the smallest gap that fits it among the buffers
/// already placed that are live with it, or after the last of them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GreedyBySize;

impl MemoryPlanner for GreedyBySize {
    fn plan(&self, requests: &[Request]) -> Option<ArenaPlan> {
        let mut order: Vec<usize> = (0..requests.len()).collect();
        order.sort_by_key(|&index| (Reverse(requests[index].size), index));
        let mut offsets = vec![ByteOffset::new(0); requests.len()];
        let mut placed: Vec<(usize, Range<u64>)> = Vec::new();
        for index in order {
            let offset = smallest_gap(requests, index, &placed)?;
            let end = offset.get().checked_add(requests[index].size.get())?;
            offsets[index] = offset;
            let at = placed.partition_point(|(_, bytes)| bytes.start <= offset.get());
            placed.insert(at, (index, offset.get()..end));
        }
        let size = placed.iter().map(|(_, bytes)| bytes.end).max().unwrap_or(0);
        Some(ArenaPlan { offsets, size: ByteSize::new(size) })
    }
}

/// The offset for request `index`: the start of the smallest gap that fits
/// it between the buffers of `placed`, sorted by offset, live with it, or
/// the first aligned offset after them.
fn smallest_gap(
    requests: &[Request],
    index: usize,
    placed: &[(usize, Range<u64>)],
) -> Option<ByteOffset> {
    let request = requests[index];
    let mut best: Option<(u64, ByteOffset)> = None;
    let mut taken_to = ByteOffset::new(0);
    for (other, bytes) in placed {
        if !live_together(request.lifetime, requests[*other].lifetime) {
            continue;
        }
        let start = taken_to.align_up(BUFFER_ALIGNMENT).ok()?;
        let end = start.get().checked_add(request.size.get())?;
        if end <= bytes.start && best.is_none_or(|(gap, _)| bytes.start - start.get() < gap) {
            best = Some((bytes.start - start.get(), start));
        }
        taken_to = taken_to.max(ByteOffset::new(bytes.end));
    }
    match best {
        Some((_, start)) => Some(start),
        None => taken_to.align_up(BUFFER_ALIGNMENT).ok(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{GreedyBySize, smallest_gap};
    use crate::memory_plan::lifetime::Lifetime;
    use crate::memory_plan::planner::{MemoryPlanner, Request};
    use ptxc_core::{ByteOffset, ByteSize};

    fn request(size: u64, steps: Option<(usize, usize)>) -> Request {
        Request {
            size: ByteSize::new(size),
            lifetime: steps.map(|(first, last)| Lifetime::between(first, last)),
        }
    }

    #[test]
    fn buffers_never_live_together_share_their_bytes() {
        let requests = [request(1000, Some((0, 1))), request(600, Some((2, 3)))];

        let plan = GreedyBySize.plan(&requests).unwrap();

        assert_eq!(plan.offsets, [ByteOffset::new(0), ByteOffset::new(0)]);
        assert_eq!(plan.size, ByteSize::new(1000));
    }

    #[test]
    fn buffers_live_together_are_placed_largest_first() {
        let requests = [request(100, Some((0, 1))), request(600, Some((1, 2)))];

        let plan = GreedyBySize.plan(&requests).unwrap();

        assert_eq!(plan.offsets, [ByteOffset::new(768), ByteOffset::new(0)]);
        assert_eq!(plan.size, ByteSize::new(868));
    }

    #[test]
    fn a_buffer_takes_the_smallest_gap_that_fits_it() {
        let requests = [
            request(256, Some((0, 9))),
            request(256, Some((0, 9))),
            request(256, Some((0, 9))),
            request(256, Some((5, 5))),
        ];
        let placed = [(0, 0..256), (1, 1280..1536), (2, 1792..2048)];

        assert_eq!(smallest_gap(&requests, 3, &placed), Some(ByteOffset::new(1536)));
    }

    #[test]
    fn a_smaller_gap_before_a_larger_one_keeps_the_buffer() {
        let requests = [
            request(1024, Some((0, 9))),
            request(256, Some((0, 9))),
            request(256, Some((0, 9))),
            request(256, Some((5, 5))),
        ];
        let placed = [(0, 0..1024), (1, 1280..1536), (2, 2048..2304)];

        assert_eq!(smallest_gap(&requests, 3, &placed), Some(ByteOffset::new(1024)));
    }

    #[test]
    fn of_gaps_that_fit_equally_well_a_buffer_takes_the_lowest() {
        let requests = [
            request(256, Some((0, 9))),
            request(256, Some((0, 9))),
            request(256, Some((0, 9))),
            request(256, Some((5, 5))),
        ];
        let placed = [(0, 0..256), (1, 512..768), (2, 1024..1280)];

        assert_eq!(smallest_gap(&requests, 3, &placed), Some(ByteOffset::new(256)));
    }

    #[test]
    fn a_buffer_no_gap_fits_goes_after_the_buffers_live_with_it() {
        let requests =
            [request(256, Some((0, 9))), request(256, Some((0, 9))), request(300, Some((5, 5)))];
        let placed = [(0, 0..256), (1, 512..768)];

        assert_eq!(smallest_gap(&requests, 2, &placed), Some(ByteOffset::new(768)));
    }

    #[test]
    fn the_bytes_of_a_buffer_not_live_with_it_are_free() {
        let requests =
            [request(256, Some((0, 1))), request(256, Some((0, 9))), request(256, Some((5, 5)))];
        let placed = [(0, 0..256), (1, 256..512)];

        assert_eq!(smallest_gap(&requests, 2, &placed), Some(ByteOffset::new(0)));
    }
}
