//! `LoopTree`: a compute definition's loop nest as a schedule has shaped it.

use super::error::ScheduleError;
use super::primitive::ThreadAxis;
use crate::compute::{ComputeDef, IterKind, IterVarId, VarName};
use crate::expr::{ExprArena, ExprId, Interval, Simplifier, VarId, VarRanges};
use ptxc_core::{Dim, Idx, IndexVec};

ptxc_core::define_index!(
    /// A loop a schedule created. Ids are never reused, so a trace replayed
    /// on the same definition makes the same ids.
    pub LoopId
);

/// Whether a loop ranges over output elements or over a reduction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LoopKind {
    /// Made from spatial axes.
    Spatial,
    /// Made from reduction axes.
    Reduce,
}

/// How a loop's iterations run.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LoopAnnotation {
    /// One after another.
    Serial,
    /// Fully unrolled.
    Unrolled,
    /// As one vector operation.
    Vectorized,
    /// In parallel, on a grid or block axis.
    Bound(ThreadAxis),
}

/// A loop: its name, extent, kind and how it runs.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Loop {
    /// The name it prints with: the axis's, with `.0`, `.1` for split
    /// parts and `_` joining fused names.
    pub name: VarName,
    /// The number of iterations.
    pub extent: Dim,
    /// Spatial or reduction.
    pub kind: LoopKind,
    /// How the iterations run.
    pub annotation: LoopAnnotation,
}

/// A condition the loop nest must check: iterations where `index` reaches
/// `extent` fall outside the iteration domain, which a split by a factor
/// that does not divide the extent makes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Guard {
    /// The index, over loop variables.
    pub index: ExprId,
    /// The bound it must stay below.
    pub extent: Dim,
}

/// The loop nest of one compute definition: every loop made so far, the
/// current nest from outermost to innermost, each axis's index over the
/// nest's loop variables, and the guards the nest needs.
#[derive(Clone, Debug)]
pub struct LoopTree {
    pub(super) loops: IndexVec<LoopId, Loop>,
    pub(super) nest: Vec<LoopId>,
    pub(super) exprs: ExprArena,
    pub(super) axes: IndexVec<IterVarId, ExprId>,
    pub(super) guards: Vec<Guard>,
}

impl LoopTree {
    /// One serial loop per axis: the spatial axes outermost, then the
    /// reduction axes, each in declaration order.
    pub(crate) fn new(compute: &ComputeDef) -> Self {
        let mut tree = Self {
            loops: IndexVec::new(),
            nest: Vec::new(),
            exprs: ExprArena::new(),
            axes: IndexVec::new(),
            guards: Vec::new(),
        };
        let mut reductions = Vec::new();
        for iter in compute.iter_vars() {
            let kind = match iter.kind {
                IterKind::Spatial => LoopKind::Spatial,
                IterKind::Reduce => LoopKind::Reduce,
            };
            let id = tree.add_loop(iter.name.clone(), iter.extent, kind);
            let index = tree.var(id);
            let _axis: IterVarId = tree.axes.push(index);
            if kind == LoopKind::Reduce { reductions.push(id) } else { tree.nest.push(id) }
        }
        tree.nest.extend(reductions);
        tree
    }

    /// Every loop made, current or not.
    pub const fn loops(&self) -> &IndexVec<LoopId, Loop> {
        &self.loops
    }

    /// The current nest, outermost first.
    pub fn nest(&self) -> &[LoopId] {
        &self.nest
    }

    /// The arena the axis indices and guards live in; loop `l`'s variable
    /// is `VarId` `l`.
    pub const fn exprs(&self) -> &ExprArena {
        &self.exprs
    }

    /// The index of `axis`, over the nest's loop variables.
    pub fn axis_index(&self, axis: IterVarId) -> ExprId {
        self.axes[axis]
    }

    /// The guards the nest must check.
    pub fn guards(&self) -> &[Guard] {
        &self.guards
    }

    /// Each current loop's variable ranges over `[0, extent)`.
    pub fn ranges(&self) -> VarRanges {
        let mut ranges = VarRanges::new();
        for &id in &self.nest {
            let last = self.loops[id]
                .extent
                .get()
                .checked_sub(1)
                .and_then(|last| i64::try_from(last).ok());
            if let Some(range) = last.and_then(|last| Interval::new(0, last)) {
                ranges.insert(Self::var_of(id), range);
            }
        }
        ranges
    }

    pub(super) fn var_of(id: LoopId) -> VarId {
        VarId::new(id.index())
    }

    pub(super) fn var(&mut self, id: LoopId) -> ExprId {
        self.exprs.var(Self::var_of(id))
    }

    pub(super) fn add_loop(&mut self, name: VarName, extent: Dim, kind: LoopKind) -> LoopId {
        self.loops.push(Loop { name, extent, kind, annotation: LoopAnnotation::Serial })
    }

    /// The loop `id`, which must be in the nest and still serial.
    pub(super) fn serial_loop(&self, id: LoopId) -> Result<&Loop, ScheduleError> {
        let current = self.current_loop(id)?;
        if current.annotation == LoopAnnotation::Serial {
            Ok(current)
        } else {
            Err(ScheduleError::Annotated(current.name.clone()))
        }
    }

    /// The loop `id`, which must be in the nest.
    pub(super) fn current_loop(&self, id: LoopId) -> Result<&Loop, ScheduleError> {
        let found = self.loops.get(id).ok_or(ScheduleError::UnknownLoop(id))?;
        if self.nest.contains(&id) {
            Ok(found)
        } else {
            Err(ScheduleError::NotInNest(found.name.clone()))
        }
    }

    /// The position of `id` in the nest, which it must be in.
    pub(super) fn position(&self, id: LoopId) -> Result<usize, ScheduleError> {
        let found = self.loops.get(id).ok_or(ScheduleError::UnknownLoop(id))?;
        self.nest
            .iter()
            .position(|&nested| nested == id)
            .ok_or_else(|| ScheduleError::NotInNest(found.name.clone()))
    }

    /// Replaces `old`'s variable by `value` in every axis index and guard,
    /// after `old` has left the nest, then simplifies them over the new
    /// loops' ranges. Substitution keeps the values an index takes, so a
    /// guard stays exactly as necessary as it was.
    pub(super) fn replace_var(&mut self, old: LoopId, value: ExprId) {
        let ranges = self.ranges();
        let mut simplifier = Simplifier::new(&ranges);
        let var = Self::var_of(old);
        for axis in self.axes.ids() {
            let substituted = self.exprs.substitute(self.axes[axis], var, value);
            self.axes[axis] = simplifier.simplify(&mut self.exprs, substituted);
        }
        for guard in &mut self.guards {
            let substituted = self.exprs.substitute(guard.index, var, value);
            guard.index = simplifier.simplify(&mut self.exprs, substituted);
        }
    }
}
