//! Asynchronous copies as `cp.async` makes them: a thread's copies land
//! only when a wait completes the group they were committed in, and until
//! then no thread may touch the elements they write.

use super::RunError;
use super::grid::Block;
use super::value::Value;
use crate::compute::Scalar;
use crate::expr::ExprId;
use crate::stmt::{BufferId, Condition};
use ptxc_core::Dim;
use std::collections::{HashMap, HashSet, VecDeque};

/// A copy started and not landed: where its elements go, and their values,
/// read when it started.
struct Landing {
    buffer: BufferId,
    first: i64,
    run: Dim,
    values: Vec<Value>,
}

/// The copies a block's threads have started and not landed.
#[derive(Default)]
pub(super) struct InFlight {
    /// Each thread's copies started since its last commit.
    open: HashMap<usize, Vec<Landing>>,
    /// Each thread's committed groups, oldest first.
    committed: HashMap<usize, VecDeque<Vec<Landing>>>,
    /// The elements some copy in flight writes.
    targets: HashSet<(BufferId, i64)>,
}

impl Block<'_, '_> {
    /// Starts each active thread's copy of `run` elements: read now, or
    /// zero where one of `bounds` fails, and written in this phase.
    pub(super) fn start_copies(
        &mut self,
        [(target, target_index), (source, source_index)]: [(BufferId, ExprId); 2],
        (run, bounds): (Dim, &[Condition]),
        active: &[usize],
    ) -> Result<(), RunError> {
        for &thread in active {
            let values = if self.holds(bounds, thread)? {
                let first = self.index(source_index, thread)?;
                positions(first, run)
                    .map(|position| self.load(source, position?, thread))
                    .collect::<Result<_, _>>()?
            } else {
                let zero = Value::of_scalar(Scalar::zero(self.grid.kernel.buffers()[target].dtype));
                (0..run.get()).map(|_| zero).collect()
            };
            let first = self.index(target_index, thread)?;
            self.claim(target, first, run, thread)?;
            let landing = Landing { buffer: target, first, run, values };
            self.copies.open.entry(thread).or_default().push(landing);
        }
        Ok(())
    }

    /// Closes each active thread's group of the copies it started since the
    /// last, empty if it started none.
    pub(super) fn commit_copies(&mut self, active: &[usize]) {
        for &thread in active {
            let group = self.copies.open.remove(&thread).unwrap_or_default();
            self.copies.committed.entry(thread).or_default().push_back(group);
        }
    }

    /// Lands each active thread's committed groups but the newest `pending`.
    pub(super) fn land_copies(&mut self, pending: u32, active: &[usize]) -> Result<(), RunError> {
        let pending = usize::try_from(pending).unwrap_or(usize::MAX);
        for &thread in active {
            let Some(groups) = self.copies.committed.get_mut(&thread) else { continue };
            let due = groups.len().saturating_sub(pending);
            let landings: Vec<Landing> = groups.drain(..due).flatten().collect();
            for landing in landings {
                self.land(landing, thread)?;
            }
        }
        Ok(())
    }

    /// `index` of `buffer`, unless a copy in flight writes it.
    pub(super) fn settled(&self, buffer: BufferId, index: i64) -> Result<(), RunError> {
        if self.copies.targets.contains(&(buffer, index)) {
            let name = self.grid.kernel.buffers()[buffer].name.clone();
            Err(RunError::InFlight { buffer: name, index })
        } else {
            Ok(())
        }
    }

    /// Nothing, if no copy in flight writes `buffer`, which is being freed.
    pub(super) fn all_landed(&self, buffer: BufferId) -> Result<(), RunError> {
        if self.copies.targets.iter().any(|&(written, _)| written == buffer) {
            Err(RunError::Unlanded(self.grid.kernel.buffers()[buffer].name.clone()))
        } else {
            Ok(())
        }
    }

    /// Whether every one of `bounds` holds for `thread`.
    fn holds(&self, bounds: &[Condition], thread: usize) -> Result<bool, RunError> {
        for condition in bounds {
            if self.index(condition.index, thread)? >= self.index(condition.bound, thread)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Marks `run` elements of `buffer` from `first` as written by
    /// `thread`'s copy in flight.
    fn claim(
        &mut self,
        buffer: BufferId,
        first: i64,
        run: Dim,
        thread: usize,
    ) -> Result<(), RunError> {
        let name = &self.grid.kernel.buffers()[buffer].name;
        for position in positions(first, run) {
            let position = position?;
            self.settled(buffer, position)?;
            self.phase.write(buffer, name, position, thread)?;
            let _new = self.copies.targets.insert((buffer, position));
        }
        Ok(())
    }

    fn land(&mut self, landing: Landing, thread: usize) -> Result<(), RunError> {
        for (position, value) in positions(landing.first, landing.run).zip(landing.values) {
            let position = position?;
            let _landed = self.copies.targets.remove(&(landing.buffer, position));
            self.store(landing.buffer, position, value, thread)?;
        }
        Ok(())
    }
}

/// The `run` adjacent positions from `first`.
fn positions(first: i64, run: Dim) -> impl Iterator<Item = Result<i64, RunError>> {
    (0..run.get()).map(move |offset| {
        i64::try_from(offset)
            .ok()
            .and_then(|offset| first.checked_add(offset))
            .ok_or(RunError::Index)
    })
}
