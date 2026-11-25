//! Split-K for a product whose output has too few tiles to keep the GPU
//! busy, as CUTLASS's parallel split-K splits one: its summed axis is cut
//! into slices, each slice of each tile a block's work, and a second stage
//! sums the slices (D75).

use crate::default::default_schedule;
use crate::rules::matmul_shape::{Product, product};
use crate::rules::{FULL_GRID, RESIDENT_GRID};
use ptxc_core::{ByteSize, Dim, ElementCount};
use ptxc_loop::{ComputeDef, LaunchShape, Rfactored, ScheduleError, Slices, rfactor};
use ptxc_target::{BlockCount, Target};
use std::sync::Arc;

/// What every slice's depth is a multiple of: a whole number of every
/// tiled rule's steps, so no slice ends in a partial step.
const SLICE_ALIGNMENT: u64 = 64;

/// The shallowest slice: enough steps for the deepest pipeline to fill.
const MIN_SLICE_DEPTH: u64 = 128;

/// `compute` split into slices of its summed axis, if it is a product its
/// default schedule leaves the GPU idle on and `slices` finds a split.
pub fn split_k(
    compute: &Arc<ComputeDef>,
    target: &Target,
) -> Result<Option<Rfactored>, ScheduleError> {
    let Some(product) = product(compute) else { return Ok(None) };
    let launch = default_schedule(Arc::clone(compute), target)?.lower().launch();
    let Some(slices) = workload(compute, product, launch).and_then(slices) else {
        return Ok(None);
    };
    Ok(Some(rfactor(compute, product.depth, slices)?))
}

/// What decides how a product splits.
#[derive(Clone, Copy, Debug)]
struct Workload {
    /// The blocks its default schedule launches.
    blocks: BlockCount,
    /// The extent of its summed axis.
    depth: Dim,
    /// What its two operands hold.
    operands: ByteSize,
    /// What one slice's partials move: written by the partial stage, then
    /// read by the combine.
    partials: ByteSize,
}

/// `product`'s workload under `launch`; `None` if its bytes overflow, as
/// no product that large launches too few blocks.
fn workload(compute: &ComputeDef, product: Product, launch: LaunchShape) -> Option<Workload> {
    let blocks = launch.grid.iter().try_fold(1_u32, |total, count| total.checked_mul(count.get()));
    let bytes = |tensor| {
        let input = &compute.inputs()[tensor];
        input.shape.element_count().in_bytes(input.dtype).ok().map(ByteSize::get)
    };
    let operands = bytes(product.lhs)?.checked_add(bytes(product.rhs)?)?;
    let [output] = compute.outputs() else { return None };
    let sum = compute.dtype_of(output.expr.reduction()?);
    let partial = ElementCount::new(compute.output_shape().element_count().get());
    let partials = partial.in_bytes(sum).ok()?.get().checked_mul(2)?;
    Some(Workload {
        blocks: BlockCount::new(blocks?),
        depth: compute.iter_vars()[product.depth].extent,
        operands: ByteSize::new(operands),
        partials: ByteSize::new(partials),
    })
}

/// The slices `workload` splits into, if its blocks number fewer than
/// `FULL_GRID`: the fewest with which they launch `RESIDENT_GRID`, or the
/// most there can be if none do. Each slice is a multiple of
/// `SLICE_ALIGNMENT` and at least `MIN_SLICE_DEPTH` deep, and the slices'
/// partials move no more bytes than the operands hold, past which the
/// combine costs more than the split saves.
fn slices(workload: Workload) -> Option<Slices> {
    let blocks = u64::from(workload.blocks.get());
    if blocks >= FULL_GRID {
        return None;
    }
    let depth = workload.depth.get();
    let affordable = workload.operands.get().checked_div(workload.partials.get())?;
    let fits = |&count: &u64| {
        let divides = depth.is_multiple_of(count);
        let aligned = (depth / count).is_multiple_of(SLICE_ALIGNMENT);
        divides && aligned && count <= affordable
    };
    let counts: Vec<u64> = (2..=depth / MIN_SLICE_DEPTH).filter(fits).collect();
    let fills = counts.iter().copied().find(|count| blocks * count >= RESIDENT_GRID);
    Slices::new(fills.or_else(|| counts.last().copied())?)
}

#[cfg(test)]
mod tests {
    use super::{Workload, slices};
    use ptxc_core::{ByteSize, Dim};
    use ptxc_target::BlockCount;

    /// The slices of a product launching `blocks` blocks over `depth`, its
    /// operands and each slice's partials of the given bytes.
    fn count(blocks: u32, depth: u64, [operands, partials]: [u64; 2]) -> Option<u64> {
        let workload = Workload {
            blocks: BlockCount::new(blocks),
            depth: Dim::new(depth),
            operands: ByteSize::new(operands),
            partials: ByteSize::new(partials),
        };
        slices(workload).map(ptxc_loop::Slices::get)
    }

    /// Operands far larger than any slice's partials.
    const AMPLE: [u64; 2] = [1 << 30, 1];

    #[test]
    fn a_grid_that_fills_the_gpu_is_not_split() {
        assert_eq!(count(32, 4096, AMPLE), None);
        assert_eq!(count(31, 4096, AMPLE), Some(4));
    }

    #[test]
    fn the_fewest_slices_that_fill_the_gpu_are_taken() {
        assert_eq!(count(12, 3072, AMPLE), Some(8));
        assert_eq!(count(24, 3072, AMPLE), Some(4));
    }

    #[test]
    fn where_no_slices_fill_the_gpu_the_most_there_can_be_are_taken() {
        // 768 splits into at most 6 slices of 128; 4096 into 32.
        assert_eq!(count(12, 768, AMPLE), Some(6));
        assert_eq!(count(1, 4096, AMPLE), Some(32));
    }

    #[test]
    fn slices_stay_aligned_and_deep_enough() {
        // No multiple of 64 divides 1000 into two or more slices, nor 320
        // into two of at least 128.
        assert_eq!(count(1, 1000, AMPLE), None);
        assert_eq!(count(1, 320, AMPLE), None);
        assert_eq!(count(1, 255, AMPLE), None);
        assert_eq!(count(1, 256, AMPLE), Some(2));
    }

    #[test]
    fn partials_may_move_no_more_bytes_than_the_operands_hold() {
        // A 128 by 768 by 768 f16 product: 1.4 MB of operands, and 786 KB
        // of partials per slice, written and read.
        assert_eq!(count(24, 768, [1_376_256, 786_432]), None);
        assert_eq!(count(12, 3072, [800, 200]), Some(4));
    }
}
