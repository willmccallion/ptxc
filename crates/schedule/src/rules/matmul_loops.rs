//! The loops a matrix product's tilings share, on CUDA cores and tensor
//! cores alike: its rows, columns and depth as one loop each, with its
//! batch loops; a loop split into parts; the block loops fused onto the
//! grid; and both operands staged at a reduction step that is pipelined.

use super::matmul_shape::Product;
use super::{bind, fused};
use crate::pipeline::stages_for;
use crate::tiling::StepTile;
use ptxc_loop::{
    IterVarId, LoopId, LoopKind, Open, Primitive, Schedule, ScheduleError, SplitFactor,
    TensorParamId, ThreadAxis,
};
use ptxc_target::Target;

/// A matrix product's loops: one each for its rows, columns and depth, the
/// spatial loops besides, in nest order, and its two operands.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct GemmLoops {
    pub(crate) row: LoopId,
    pub(crate) column: LoopId,
    pub(crate) depth: LoopId,
    pub(crate) batch: Vec<LoopId>,
    pub(crate) operands: [TensorParamId; 2],
}

/// `product`'s loops in the default nest: its axes' own loops.
pub(crate) fn product_loops(
    schedule: &Schedule<Open>,
    product: Product,
) -> Result<GemmLoops, ScheduleError> {
    let [row, column, depth] =
        [product.row, product.column, product.depth].map(|axis| own_loop(schedule, axis));
    let (row, column, depth) = (row?, column?, depth?);
    let tree = schedule.loops();
    let batch = tree
        .nest()
        .iter()
        .copied()
        .filter(|&id| tree.loops()[id].kind == LoopKind::Spatial && id != row && id != column)
        .collect();
    Ok(GemmLoops { row, column, depth, batch, operands: [product.lhs, product.rhs] })
}

/// `id` split by `factors`, its parts outermost first.
pub(super) fn parts<const PARTS: usize>(
    schedule: &mut Schedule<Open>,
    id: LoopId,
    factors: &[u64],
) -> Result<[LoopId; PARTS], ScheduleError> {
    let factors: Option<Vec<SplitFactor>> =
        factors.iter().map(|&factor| SplitFactor::new(factor)).collect();
    let split = schedule.split(id, factors.ok_or(ScheduleError::EmptySplit)?)?;
    <[LoopId; PARTS]>::try_from(split).map_err(|_| ScheduleError::EmptySplit)
}

/// The loop of `axis` alone, as every axis has in the default nest.
pub(super) fn own_loop(
    schedule: &Schedule<Open>,
    axis: IterVarId,
) -> Result<LoopId, ScheduleError> {
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

/// `step`, at which both `operands` are staged in `slices`, pipelined over
/// the slices' stages where they have them, or as many as fit shared
/// memory if fewer.
pub(super) fn pipeline_step(
    schedule: &mut Schedule<Open>,
    operands: [TensorParamId; 2],
    (step, slices): (LoopId, StepTile),
    target: &Target,
) -> Result<(), ScheduleError> {
    let Some(wanted) = slices.stages else { return Ok(()) };
    let size = |tensor| schedule.compute().inputs()[tensor].dtype.element_size();
    let bytes = slices.bytes(operands.map(size));
    if let Some(stages) = stages_for(bytes, wanted, target) {
        schedule.apply(Primitive::Pipeline { loop_: step, stages })?;
    }
    Ok(())
}

/// Both `operands` staged in shared memory at `step`.
pub(super) fn stage_operands(
    schedule: &mut Schedule<Open>,
    operands: [TensorParamId; 2],
    step: LoopId,
) -> Result<(), ScheduleError> {
    for tensor in operands {
        schedule.apply(Primitive::CacheRead { tensor, at: step })?;
    }
    Ok(())
}
