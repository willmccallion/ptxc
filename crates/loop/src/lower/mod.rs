//! Lowers a validated schedule to a `LoweredKernel`.

mod cache;
mod combine;
mod context;
mod cross_thread;
mod elementwise;
mod reduction;

use crate::schedule::{LoopAnnotation, LoopKind, Schedule, Validated};
use crate::stmt::{LaunchShape, LoweredKernel, MemScope};
pub(crate) use cache::tile_extents;
use context::Lowerer;
use ptxc_core::KernelName;
use ptxc_target::{BlockCount, SmemBytes, ThreadsPerBlock};

impl Schedule<Validated> {
    /// The schedule as an explicit loop nest: bound loops become grid and
    /// block indices, guards become `If`s, and a reduction is accumulated
    /// per thread and, when bound to threads, combined in shared memory.
    pub fn lower(&self) -> LoweredKernel {
        let tree = self.loops();
        let mut lowerer = Lowerer::new(self.compute(), tree);
        lowerer.stage(self.tiles(), self.threads());
        let reduction =
            self.compute().outputs().first().and_then(|output| output.expr.reduction()).cloned();
        let shared = tree.nest().iter().copied().find(|&id| {
            let found = &tree.loops()[id];
            found.kind == LoopKind::Reduce && matches!(found.annotation, LoopAnnotation::Bound(_))
        });
        let body = match (&reduction, shared) {
            (None, _) => lowerer.elementwise(),
            (Some(reduction), None) => lowerer.reduction(reduction),
            (Some(reduction), Some(shared)) => lowerer.cross_thread(reduction, shared),
        };
        let smem = shared_bytes(&lowerer);
        let launch = launch_shape(tree);
        LoweredKernel {
            name: KernelName::new(self.compute().name().as_str()),
            params: lowerer.params,
            buffers: lowerer.buffers,
            vars: lowerer.vars,
            launch,
            smem,
            body,
            exprs: lowerer.exprs,
        }
    }
}

/// The bytes of the shared buffers.
fn shared_bytes(lowerer: &Lowerer<'_>) -> SmemBytes {
    let bytes: u64 = lowerer
        .buffers
        .iter()
        .filter(|buffer| buffer.scope == MemScope::Shared)
        .map(|buffer| buffer.size.get().saturating_mul(buffer.dtype.element_size().get()))
        .sum();
    SmemBytes::new(bound_extent(bytes))
}

/// `extent` as a `u32`. Validation bounds every bound loop's extent, and so
/// the block's threads and its shared memory (at most 12 KiB), by the
/// target's limits, which are `u32`s.
pub(super) fn bound_extent(extent: u64) -> u32 {
    u32::try_from(extent).unwrap_or(u32::MAX)
}

fn launch_shape(tree: &crate::schedule::LoopTree) -> LaunchShape {
    use crate::schedule::ThreadAxis::{BlockX, BlockY, BlockZ, ThreadX, ThreadY, ThreadZ};
    let count = |axis| bound_extent(tree.axis_extent(axis));
    LaunchShape {
        grid: [BlockX, BlockY, BlockZ].map(|axis| BlockCount::new(count(axis))),
        block: [ThreadX, ThreadY, ThreadZ].map(|axis| ThreadsPerBlock::new(count(axis))),
    }
}
