//! Live ranges of buffers over the step order, and the bytes live at once.

use ptxc_core::{ByteSize, Idx, IndexVec, define_index};
use ptxc_program::{BufferId, Step};

define_index!(
    /// A step's position in a program's order.
    pub(crate) StepIndex
);

/// The steps a buffer is live over, from the first that touches it to the
/// last, both included: a step may read one buffer while it writes another,
/// so two buffers it touches are live together.
// NOTE: XLA's BufferAssignment lets an elementwise op write over an operand
// it reads for the last time. ptxc never shares a buffer within a step: a
// tiled or fused kernel's threads read elements other threads write.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Lifetime {
    first: StepIndex,
    last: StepIndex,
}

impl Lifetime {
    /// Whether some step is in both `self` and `other`.
    pub(crate) fn overlaps(self, other: Self) -> bool {
        self.first <= other.last && other.first <= self.last
    }

    /// The first step touching the buffer.
    pub(crate) const fn first(self) -> StepIndex {
        self.first
    }

    /// The last step touching the buffer.
    pub(crate) const fn last(self) -> StepIndex {
        self.last
    }

    /// `self` stretched to include `step`, a step after its first.
    const fn through(self, step: StepIndex) -> Self {
        Self { first: self.first, last: step }
    }

    /// The steps from `first` to `last`, by position.
    #[cfg(test)]
    pub(crate) fn between(first: usize, last: usize) -> Self {
        Self { first: StepIndex::new(first), last: StepIndex::new(last) }
    }
}

/// Whether two buffers with these lifetimes are live at some step together;
/// a buffer no step touches is never live.
pub(crate) fn live_together(first: Option<Lifetime>, second: Option<Lifetime>) -> bool {
    match (first, second) {
        (Some(first), Some(second)) => first.overlaps(second),
        _ => false,
    }
}

/// The lifetime of each of `buffers`, `None` for one `steps` never touch.
pub(crate) fn lifetimes<T>(
    steps: &[Step],
    buffers: &IndexVec<BufferId, T>,
) -> IndexVec<BufferId, Option<Lifetime>> {
    let mut found: IndexVec<BufferId, Option<Lifetime>> = buffers.iter().map(|_| None).collect();
    for (position, step) in steps.iter().enumerate() {
        let step_index = StepIndex::new(position);
        for buffer in touched(step) {
            if let Some(lifetime) = found.get_mut(buffer) {
                let fresh = Lifetime { first: step_index, last: step_index };
                *lifetime = Some(lifetime.map_or(fresh, |earlier| earlier.through(step_index)));
            }
        }
    }
    found
}

/// The buffers `step` reads or writes.
fn touched(step: &Step) -> Vec<BufferId> {
    match step {
        Step::Launch { args, .. } => args.clone(),
        Step::Copy { from, to } => vec![*from, *to],
    }
}

/// The most bytes `buffers`, each a size and a lifetime, hold live at any
/// one step: no placement fits them in less. `None` if that passes 2^64.
pub(crate) fn peak_live_bytes(
    buffers: impl IntoIterator<Item = (ByteSize, Lifetime)>,
) -> Option<ByteSize> {
    let mut changes: Vec<(StepIndex, Change)> = Vec::new();
    for (size, lifetime) in buffers {
        changes.push((lifetime.first, Change::Live(size)));
        changes.push((StepIndex::new(lifetime.last.index() + 1), Change::Dead(size)));
    }
    // Bytes freed at a step are counted before bytes taken at it.
    changes.sort_unstable_by_key(|&(step, change)| (step, matches!(change, Change::Live(_))));
    let (mut live, mut peak) = (0_u64, 0_u64);
    for (_, change) in changes {
        live = match change {
            Change::Live(size) => live.checked_add(size.get())?,
            Change::Dead(size) => live - size.get(),
        };
        peak = peak.max(live);
    }
    Some(ByteSize::new(peak))
}

/// A buffer becoming live, or dead, at a step.
#[derive(Clone, Copy)]
enum Change {
    Live(ByteSize),
    Dead(ByteSize),
}

#[cfg(test)]
mod tests {
    use super::{Lifetime, lifetimes, peak_live_bytes};
    use ptxc_core::{ByteSize, Idx, IndexVec};
    use ptxc_program::{BufferId, KernelId, Step};

    fn launch(args: &[usize]) -> Step {
        Step::Launch {
            kernel: KernelId::new(0),
            args: args.iter().map(|&a| BufferId::new(a)).collect(),
        }
    }

    #[test]
    fn a_buffer_lives_from_the_first_step_touching_it_to_the_last() {
        let steps = [
            launch(&[0]),
            launch(&[0, 1]),
            Step::Copy { from: BufferId::new(1), to: BufferId::new(2) },
        ];
        let buffers: IndexVec<BufferId, ()> = (0..4).map(|_| ()).collect();

        let found = lifetimes(&steps, &buffers);

        let expected = [
            Some(Lifetime::between(0, 1)),
            Some(Lifetime::between(1, 2)),
            Some(Lifetime::between(2, 2)),
            None,
        ];
        assert_eq!(found.iter().copied().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn lifetimes_overlap_when_they_share_a_step() {
        assert!(Lifetime::between(0, 2).overlaps(Lifetime::between(2, 3)));
        assert!(Lifetime::between(1, 1).overlaps(Lifetime::between(0, 4)));
        assert!(!Lifetime::between(0, 1).overlaps(Lifetime::between(2, 3)));
        assert!(!Lifetime::between(2, 3).overlaps(Lifetime::between(0, 1)));
    }

    #[test]
    fn the_peak_is_the_most_bytes_live_at_one_step() {
        let buffers = [
            (ByteSize::new(100), Lifetime::between(0, 1)),
            (ByteSize::new(30), Lifetime::between(1, 2)),
            (ByteSize::new(50), Lifetime::between(2, 3)),
            (ByteSize::new(70), Lifetime::between(3, 3)),
        ];

        assert_eq!(peak_live_bytes(buffers), Some(ByteSize::new(130)));
    }

    #[test]
    fn a_buffer_dead_after_a_step_frees_its_bytes_for_the_next() {
        let buffers = [
            (ByteSize::new(100), Lifetime::between(0, 0)),
            (ByteSize::new(100), Lifetime::between(1, 1)),
        ];

        assert_eq!(peak_live_bytes(buffers), Some(ByteSize::new(100)));
    }

    #[test]
    fn no_buffers_need_no_bytes() {
        assert_eq!(peak_live_bytes([]), Some(ByteSize::new(0)));
    }

    #[test]
    fn live_bytes_past_2_to_the_64_have_no_peak() {
        let huge = ByteSize::new(u64::MAX / 2 + 1);

        assert_eq!(
            peak_live_bytes([(huge, Lifetime::between(0, 1)), (huge, Lifetime::between(1, 2))]),
            None
        );
    }
}
