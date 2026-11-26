//! The `MemoryPlanner` trait: offsets in one arena for buffers with
//! lifetimes.

use super::lifetime::Lifetime;
use ptxc_core::{ByteOffset, ByteSize};

/// A buffer to place in the arena.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Request {
    /// Its size.
    pub(crate) size: ByteSize,
    /// When it is live; `None` if no step touches it.
    pub(crate) lifetime: Option<Lifetime>,
}

/// A plan: an offset for each buffer and the arena holding them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct ArenaPlan {
    /// Each request's offset, in the order of the requests.
    pub(crate) offsets: Vec<ByteOffset>,
    /// The arena's size: the end of the buffer ending last.
    pub(crate) size: ByteSize,
}

/// Places buffers in one arena so that two live at the same step never
/// share a byte, each at an offset aligned for any buffer.
pub(crate) trait MemoryPlanner {
    /// A plan for `requests`; `None` if an offset passes 2^64.
    fn plan(&self, requests: &[Request]) -> Option<ArenaPlan>;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{MemoryPlanner, Request};
    use crate::memory_plan::greedy_by_size::GreedyBySize;
    use crate::memory_plan::lifetime::{Lifetime, peak_live_bytes};
    use crate::memory_plan::verify::{Placed, verify};
    use crate::memory_plan::{PLANNERS, smallest_plan};
    use proptest::prelude::*;
    use ptxc_core::{ByteOffset, ByteSize, Idx};
    use ptxc_program::{BUFFER_ALIGNMENT, BufferId};

    #[test]
    fn the_smallest_plan_is_linear_scans_where_greedy_by_size_strands_a_gap() {
        let request = |size, first, last| Request {
            size: ByteSize::new(size),
            lifetime: Some(Lifetime::between(first, last)),
        };
        let requests =
            [request(768, 0, 0), request(768, 2, 4), request(768, 1, 3), request(512, 0, 1)];

        let chosen = smallest_plan(&requests).unwrap();

        assert_eq!(GreedyBySize.plan(&requests).unwrap().size, ByteSize::new(2048));
        assert_eq!(chosen.size, ByteSize::new(1536));
        assert_eq!(chosen.offsets.iter().map(|o| o.get()).collect::<Vec<_>>(), [0, 768, 0, 768]);
    }

    /// Up to 60 requests of up to 5000 bytes, each live over a few of 40
    /// steps or never touched.
    fn requests() -> impl Strategy<Value = Vec<Request>> {
        let steps = prop::option::weighted(0.9, (0_usize..40, 0_usize..8));
        let one = (0_u64..5000, steps).prop_map(|(size, steps)| Request {
            size: ByteSize::new(size),
            lifetime: steps.map(|(first, length)| Lifetime::between(first, first + length)),
        });
        prop::collection::vec(one, 0..60)
    }

    proptest! {
        #[test]
        fn every_plan_keeps_live_buffers_apart_in_no_more_than_their_own_regions(
            requests in requests(),
        ) {
            let live = requests.iter().filter_map(|r| Some((r.size, r.lifetime?)));
            let lower_bound = peak_live_bytes(live).unwrap();
            let regions: u64 = requests
                .iter()
                .map(|r| ByteOffset::new(r.size.get()).align_up(BUFFER_ALIGNMENT).unwrap().get())
                .sum();
            for planner in PLANNERS {
                let plan = planner.plan(&requests).unwrap();

                let placed: Vec<Placed> = requests
                    .iter()
                    .zip(&plan.offsets)
                    .enumerate()
                    .map(|(index, (request, &offset))| Placed {
                        buffer: BufferId::new(index),
                        offset,
                        size: request.size,
                        lifetime: request.lifetime,
                    })
                    .collect();
                prop_assert_eq!(verify(&placed, plan.size), Ok(()));
                prop_assert!(plan.size >= lower_bound);
                prop_assert!(plan.size.get() <= regions, "{:?} > {regions}", plan.size);
            }
        }

        #[test]
        fn the_smallest_plan_is_the_smallest_any_planner_makes(requests in requests()) {
            let sizes = PLANNERS.map(|planner| planner.plan(&requests).unwrap().size);

            let chosen = smallest_plan(&requests).unwrap();

            prop_assert_eq!(Some(chosen.size), sizes.into_iter().min());
        }
    }
}
