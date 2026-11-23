//! `Trace`: the replayable list of primitives, which is also the tuning record.

use super::primitive::Primitive;

/// The primitives applied to a schedule, in order. Replaying them on the
/// same compute definition rebuilds the same loop tree, loop ids included,
/// as TVM MetaSchedule's traces do (D4).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Trace(Vec<Primitive>);

impl Trace {
    /// An empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// The steps, in the order they were applied.
    pub fn steps(&self) -> &[Primitive] {
        &self.0
    }

    pub(super) fn push(&mut self, step: Primitive) {
        self.0.push(step);
    }
}
