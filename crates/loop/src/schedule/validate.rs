//! Checks an open schedule against a target: the launch fits, vector loops
//! have a vector width, a reduction shared between threads has the shape
//! lowering builds a shared-memory tree for, and staged inputs sit where
//! lowering can fill them.

use super::error::ScheduleError;
use super::loop_tree::{LoopAnnotation, LoopId, LoopKind, LoopTree};
use super::primitive::ThreadAxis;
use super::validate_caches::check_caches;
use super::validate_tensorize::{TensorizedProduct, check_tensorized};
use super::{Open, Schedule};
use ptxc_core::Dim;
use ptxc_target::{Target, ThreadsPerBlock};

/// A schedule checked against a target, ready to lower, with what the
/// checks established that lowering builds on.
#[derive(Clone, Debug)]
pub struct Validated {
    /// The threads in one block.
    threads: ThreadsPerBlock,
    /// Each cache's tile extents, per axis of its input.
    tiles: Vec<Vec<Dim>>,
    /// The product a tensorized nest computes.
    product: Option<TensorizedProduct>,
}

/// The lanes of a warp, which a tensorized nest puts on `threadIdx.x`.
pub(crate) const WARP_LANES: u64 = 32;

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
        let threads = check_launch(&self.loops, target)?;
        check_vectors(&self.loops)?;
        check_cross_thread(&self.loops)?;
        let tiles = check_caches(&self.compute, &self.loops, target)?;
        let product = check_tensorized(&self.compute, &self.loops, (&tiles, threads))?;
        Ok(Schedule {
            compute: self.compute,
            loops: self.loops,
            trace: self.trace,
            state: Validated { threads, tiles, product },
        })
    }
}

impl Schedule<Validated> {
    /// The threads in one block.
    pub(crate) const fn threads(&self) -> ThreadsPerBlock {
        self.state.threads
    }

    /// Each cache's tile extents, per axis of its input, in the order the
    /// caches were staged.
    pub(crate) fn tiles(&self) -> &[Vec<Dim>] {
        &self.state.tiles
    }

    /// The product a tensorized nest computes.
    pub(crate) const fn product(&self) -> Option<&TensorizedProduct> {
        self.state.product.as_ref()
    }
}

impl LoopTree {
    /// The extent of `axis`: the loop bound to it, a tensorized nest's
    /// warp lanes on `threadIdx.x`, or 1.
    pub(crate) fn axis_extent(&self, axis: ThreadAxis) -> u64 {
        if axis == ThreadAxis::ThreadX && self.tensorized.is_some() {
            return WARP_LANES;
        }
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

/// The threads in one block, if every axis and the block fit `target`.
fn check_launch(tree: &LoopTree, target: &Target) -> Result<ThreadsPerBlock, ScheduleError> {
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
    let limit = limits.block.threads;
    match u32::try_from(threads) {
        Ok(count) if count <= limit.get() => Ok(ThreadsPerBlock::new(count)),
        _ => Err(ScheduleError::TooManyThreads { threads, limit }),
    }
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
