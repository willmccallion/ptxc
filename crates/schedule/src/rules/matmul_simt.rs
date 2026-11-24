//! Matrix multiplication on CUDA cores, tiled as CUTLASS's SIMT GEMM is:
//! each block computes a tile of the output and each thread a smaller tile
//! of it in registers, and every step of the reduction stages the block's
//! slices of both operands in shared memory.

use super::matmul_shape::Product;
use super::{bind, fused};
use crate::tiling::{BlockTile, TileShape};
use ptxc_loop::{
    IterVarId, LoopId, LoopKind, Open, Primitive, Schedule, ScheduleError, SplitFactor, ThreadAxis,
};

/// Hand-picked until the search tunes it: 256 threads, each accumulating a
/// 4 by 4 tile, over a 64 by 64 block tile staged 8 deep, so an f32 step
/// stages 4 KiB.
const TILE: BlockTile = BlockTile {
    threads: TileShape { rows: 16, columns: 16 },
    per_thread: TileShape { rows: 4, columns: 4 },
    depth: 8,
};

/// Tiles `product`'s nest: the batch axes and the block rows and columns
/// fused onto `blockIdx.x`, which no grid limit binds; the block's threads
/// on `threadIdx.y` and `threadIdx.x`, a thread's columns innermost; and
/// the reduction split into staged steps, inside which a thread sums its
/// tile with the step's depth and its own rows and columns unrolled.
pub(crate) fn tiled_on_cuda_cores(
    schedule: &mut Schedule<Open>,
    product: Product,
) -> Result<(), ScheduleError> {
    let batch = batch_loops(schedule, product);
    let [row_block, row_thread, row_own] =
        parts(schedule, product.row, &[TILE.threads.rows, TILE.per_thread.rows])?;
    let [column_block, column_thread, column_own] =
        parts(schedule, product.column, &[TILE.threads.columns, TILE.per_thread.columns])?;
    let [step, step_depth] = parts(schedule, product.depth, &[TILE.depth])?;
    let mut order = batch.clone();
    order.extend([row_block, column_block, row_thread, column_thread]);
    order.extend([step, step_depth, row_own, column_own]);
    schedule.apply(Primitive::Reorder { order })?;
    let mut blocks = batch;
    blocks.extend([row_block, column_block]);
    if let Some(grid) = fused(schedule, &blocks)? {
        bind(schedule, grid, ThreadAxis::BlockX)?;
    }
    bind(schedule, row_thread, ThreadAxis::ThreadY)?;
    bind(schedule, column_thread, ThreadAxis::ThreadX)?;
    for tensor in [product.lhs, product.rhs] {
        schedule.apply(Primitive::CacheRead { tensor, at: step })?;
    }
    for loop_ in [step_depth, row_own, column_own] {
        schedule.apply(Primitive::Unroll { loop_ })?;
    }
    Ok(())
}

/// The spatial loops other than the row's and the column's, in nest order.
fn batch_loops(schedule: &Schedule<Open>, product: Product) -> Vec<LoopId> {
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
fn parts<const PARTS: usize>(
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
