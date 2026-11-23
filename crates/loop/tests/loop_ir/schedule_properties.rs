//! A schedule never changes what is computed: whatever splits, fuses and
//! reorders shape the nest, it visits every point of the iteration domain
//! exactly once.

use crate::schedule_support::{domain, open, reduction, visited};
use proptest::prelude::*;
use ptxc_loop::{Primitive, SplitFactor};

/// A primitive on whichever loops are at the given nest positions when it
/// is applied, so random steps stay meaningful as the nest changes.
#[derive(Clone, Debug)]
pub(crate) enum Step {
    Split { position: usize, factors: Vec<u64> },
    Fuse { position: usize, count: usize },
    Reorder { positions: Vec<usize> },
}

pub(crate) fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        (0_usize..8, prop::collection::vec(1_u64..=5, 1..=2))
            .prop_map(|(position, factors)| Step::Split { position, factors }),
        (0_usize..8, 2_usize..=3).prop_map(|(position, count)| Step::Fuse { position, count }),
        prop::collection::vec(0_usize..8, 2..=4).prop_map(|positions| Step::Reorder { positions }),
    ]
}

/// The step as a primitive on the current nest, if its positions exist.
pub(crate) fn primitive(step: &Step, nest: &[ptxc_loop::LoopId]) -> Option<Primitive> {
    let at = |position: usize| nest.get(position).copied();
    match step {
        Step::Split { position, factors } => Some(Primitive::Split {
            loop_: at(*position)?,
            factors: factors.iter().map(|&factor| SplitFactor::new(factor).unwrap()).collect(),
        }),
        Step::Fuse { position, count } => Some(Primitive::Fuse {
            loops: (*position..position + count).map(at).collect::<Option<_>>()?,
        }),
        Step::Reorder { positions } => Some(Primitive::Reorder {
            order: positions.iter().map(|&p| at(p)).collect::<Option<_>>()?,
        }),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn every_schedule_visits_each_point_once(
        extents in [1_u64..=6, 1_u64..=6, 1_u64..=5],
        steps in prop::collection::vec(step(), 0..8),
    ) {
        let compute = reduction(extents);
        let (mut schedule, _) = open(&compute);
        for step in &steps {
            let nest = schedule.loops().nest().to_vec();
            if let Some(primitive) = primitive(step, &nest) {
                let _rejected_or_applied = schedule.apply(primitive);
            }
        }

        let mut points = visited(&schedule);
        points.sort();
        prop_assert_eq!(points, domain(&extents));
    }
}
