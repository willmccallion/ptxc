//! Matrix multiplication on CUDA cores, tiled as CUTLASS's SIMT GEMM is:
//! each block computes a tile of the output and each thread a smaller tile
//! of it in registers, and every step of the reduction stages the block's
//! slices of both operands in shared memory.

use super::bind;
use super::matmul_loops::{batch_loops, bind_blocks, parts, stage_operands};
use super::matmul_shape::Product;
use crate::tiling::{BlockTile, TileShape};
use ptxc_loop::{Open, Primitive, Schedule, ScheduleError, ThreadAxis};

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
    bind_blocks(schedule, &batch, [row_block, column_block])?;
    bind(schedule, row_thread, ThreadAxis::ThreadY)?;
    bind(schedule, column_thread, ThreadAxis::ThreadX)?;
    stage_operands(schedule, product, step)?;
    for loop_ in [step_depth, row_own, column_own] {
        schedule.apply(Primitive::Unroll { loop_ })?;
    }
    Ok(())
}
