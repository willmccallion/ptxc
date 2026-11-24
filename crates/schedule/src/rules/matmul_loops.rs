//! The loops a matrix product's tilings share, on CUDA cores and tensor
//! cores alike: its batch loops, an axis's loop split into parts, the block
//! loops fused onto the grid, and both operands staged at a reduction step.

use super::matmul_shape::Product;
use super::{bind, fused};
use ptxc_loop::{
    IterVarId, LoopId, LoopKind, Open, Primitive, Schedule, ScheduleError, SplitFactor, ThreadAxis,
};

/// The spatial loops other than the row's and the column's, in nest order.
pub(super) fn batch_loops(schedule: &Schedule<Open>, product: Product) -> Vec<LoopId> {
    let tree = schedule.loops();
    let own = |axis| tree.loop_of(axis);
    let matrix = [own(product.row), own(product.column)];
    tree.nest()
        .iter()
        .copied()
        .filter(|&id| tree.loops()[id].kind == LoopKind::Spatial && !matrix.contains(&Some(id)))
        .collect()
}

/// `axis`'s own loop split by `factors`, its parts outermost first.
pub(super) fn parts<const PARTS: usize>(
    schedule: &mut Schedule<Open>,
    axis: IterVarId,
    factors: &[u64],
) -> Result<[LoopId; PARTS], ScheduleError> {
    let id = own_loop(schedule, axis)?;
    let factors: Option<Vec<SplitFactor>> =
        factors.iter().map(|&factor| SplitFactor::new(factor)).collect();
    let split = schedule.split(id, factors.ok_or(ScheduleError::EmptySplit)?)?;
    <[LoopId; PARTS]>::try_from(split).map_err(|_| ScheduleError::EmptySplit)
}

/// The loop of `axis` alone, as every axis has in the default nest.
fn own_loop(schedule: &Schedule<Open>, axis: IterVarId) -> Result<LoopId, ScheduleError> {
    schedule
        .loops()
        .loop_of(axis)
        .ok_or_else(|| ScheduleError::NotInNest(schedule.compute().iter_vars()[axis].name.clone()))
}

/// The `batch` loops and the block's row and column loops fused onto
/// `blockIdx.x`, which no grid limit binds.
pub(super) fn bind_blocks(
    schedule: &mut Schedule<Open>,
    batch: &[LoopId],
    [row_block, column_block]: [LoopId; 2],
) -> Result<(), ScheduleError> {
    let mut blocks = batch.to_vec();
    blocks.extend([row_block, column_block]);
    if let Some(grid) = fused(schedule, &blocks)? {
        bind(schedule, grid, ThreadAxis::BlockX)?;
    }
    Ok(())
}

/// Both of `product`'s operands staged in shared memory at `step`.
pub(super) fn stage_operands(
    schedule: &mut Schedule<Open>,
    product: Product,
    step: LoopId,
) -> Result<(), ScheduleError> {
    for tensor in [product.lhs, product.rhs] {
        schedule.apply(Primitive::CacheRead { tensor, at: step })?;
    }
    Ok(())
}
