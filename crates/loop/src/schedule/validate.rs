//! Checks an open schedule against a target: the launch fits, vector loops
//! have a vector width, and a reduction shared between threads has the
//! shape lowering builds a shared-memory tree for.

use super::error::ScheduleError;
use super::loop_tree::{LoopAnnotation, LoopId, LoopKind, LoopTree};
use super::primitive::ThreadAxis;
use super::{Open, Schedule};
use ptxc_target::Target;
use std::marker::PhantomData;

/// A schedule checked against a target, ready to lower.
#[derive(Clone, Copy, Debug)]
pub struct Validated;

/// Every axis in `(grid x, y, z, block x, y, z)` order.
const AXES: [ThreadAxis; 6] = [
    ThreadAxis::BlockX,
    ThreadAxis::BlockY,
    ThreadAxis::BlockZ,
    ThreadAxis::ThreadX,
    ThreadAxis::ThreadY,
    ThreadAxis::ThreadZ,
];

impl Schedule<Open> {
    /// The schedule, if `target` can run it and lowering supports its shape.
    pub fn validate(self, target: &Target) -> Result<Schedule<Validated>, ScheduleError> {
        check_launch(&self.loops, target)?;
        check_vectors(&self.loops)?;
        check_cross_thread(&self.loops)?;
        Ok(Schedule {
            compute: self.compute,
            loops: self.loops,
            trace: self.trace,
            state: PhantomData,
        })
    }
}

impl LoopTree {
    /// The extent of the loop bound to `axis`, or 1 if none is.
    pub(crate) fn axis_extent(&self, axis: ThreadAxis) -> u64 {
        self.nest
            .iter()
            .find(|&&id| self.loops[id].annotation == LoopAnnotation::Bound(axis))
            .map_or(1, |&id| self.loops[id].extent.get())
    }

    /// The threads in a block: the product of the thread axes' extents,
    /// or `None` if it overflows.
    pub(crate) fn block_threads(&self) -> Option<u64> {
        AXES[3..]
            .iter()
            .try_fold(1_u64, |threads, &axis| threads.checked_mul(self.axis_extent(axis)))
    }
}

fn check_launch(tree: &LoopTree, target: &Target) -> Result<(), ScheduleError> {
    let limits = target.limits();
    for (slot, &axis) in AXES.iter().enumerate() {
        let limit = if slot < 3 {
            u64::from(limits.grid.extents[slot].get())
        } else {
            u64::from(limits.block.extents[slot - 3].get())
        };
        let extent = tree.axis_extent(axis);
        if extent > limit {
            return Err(ScheduleError::AxisTooLarge {
                axis,
                extent: ptxc_core::Dim::new(extent),
                limit,
            });
        }
    }
    let threads = tree.block_threads().unwrap_or(u64::MAX);
    if threads > u64::from(limits.block.threads.get()) {
        return Err(ScheduleError::TooManyThreads { threads, limit: limits.block.threads });
    }
    Ok(())
}

/// A vectorized loop has 2 or 4 iterations, is the innermost serial loop,
/// and no guard reads it.
fn check_vectors(tree: &LoopTree) -> Result<(), ScheduleError> {
    let serial: Vec<LoopId> = tree
        .nest
        .iter()
        .copied()
        .filter(|&id| !matches!(tree.loops[id].annotation, LoopAnnotation::Bound(_)))
        .collect();
    for (position, &id) in serial.iter().enumerate() {
        let found = &tree.loops[id];
        if found.annotation != LoopAnnotation::Vectorized {
            continue;
        }
        if !matches!(found.extent.get(), 2 | 4) {
            return Err(ScheduleError::VectorWidth {
                name: found.name.clone(),
                extent: found.extent,
            });
        }
        if position + 1 != serial.len() {
            return Err(ScheduleError::VectorNotInnermost(found.name.clone()));
        }
        let var = LoopTree::var_of(id);
        if tree.guards.iter().any(|guard| tree.exprs.vars(guard.index).contains(&var)) {
            return Err(ScheduleError::VectorGuarded(found.name.clone()));
        }
    }
    Ok(())
}

/// A reduction shared between threads uses one thread axis and has no
/// serial spatial loop inside its serial reduction loops. Its shared memory,
/// at most 1,024 threads times a value and an `i64` position (12 KiB), fits
/// the 48 KiB every CUDA GPU gives a block without opting in.
fn check_cross_thread(tree: &LoopTree) -> Result<(), ScheduleError> {
    let reduce_axes = AXES[3..].iter().filter(|&&axis| {
        tree.nest.iter().any(|&id| {
            tree.loops[id].annotation == LoopAnnotation::Bound(axis)
                && tree.loops[id].kind == LoopKind::Reduce
        })
    });
    match reduce_axes.count() {
        0 => return Ok(()),
        1 => {}
        _ => return Err(ScheduleError::CrossThreadAxes),
    }
    let serial = tree
        .nest
        .iter()
        .filter(|&&id| !matches!(tree.loops[id].annotation, LoopAnnotation::Bound(_)));
    let mut inside_reduction = false;
    for &id in serial {
        let found = &tree.loops[id];
        inside_reduction |= found.kind == LoopKind::Reduce;
        if inside_reduction && found.kind == LoopKind::Spatial {
            return Err(ScheduleError::SpatialInsideCrossThread(found.name.clone()));
        }
    }
    Ok(())
}
