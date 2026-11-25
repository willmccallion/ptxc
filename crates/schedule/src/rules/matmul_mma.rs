//! Matrix multiplication on tensor cores, tiled as CUTLASS's tensor-op
//! GEMM is: each block computes a tile of the output, each warp a grid of
//! `mma` tiles of it held in fragments, and every step of the reduction
//! stages the block's slices of both operands in shared memory, from which
//! the warps load their fragments (D58).

use super::bind;
use super::matmul_loops::{GemmLoops, bind_blocks, parts, pipeline_step, stage_operands};
use crate::tiling::{TensorCoreTile, TileShape};
use ptxc_core::Dim;
use ptxc_loop::{
    ComputeDef, LoopId, Open, Primitive, Schedule, ScheduleError, StageCount, Swizzle, TensorExpr,
    TensorParamId, ThreadAxis,
};
use ptxc_target::{MmaInstruction, MmaShape, Target};

/// Hand-picked until the search tunes them: four warps, each holding 4 by
/// 8 `mma` tiles, over a block tile of 128 by 128 at f16's m16n8, staged
/// two `mma` steps deep, so an f16 step stages 16 KiB, its steps
/// pipelined over two stages and its tiles swizzled. Either alone is
/// slower; together they are faster on all but the smallest grids, and
/// three stages are no faster (D68)...
const LARGE: TensorCoreTile = TensorCoreTile {
    warps: TileShape { rows: 2, columns: 2 },
    per_warp: TileShape { rows: 4, columns: 8 },
    steps: 2,
    stages: StageCount::new(2),
};

/// ...and, for an output too small for enough large tiles, four warps
/// holding 2 by 4 tiles each, over a block tile of 64 by 64, its steps
/// pipelined over four stages, which its few blocks cannot hide otherwise,
/// and its tiles swizzled.
const SMALL: TensorCoreTile = TensorCoreTile {
    warps: TileShape { rows: 2, columns: 2 },
    per_warp: TileShape { rows: 2, columns: 4 },
    steps: 2,
    stages: StageCount::new(4),
};

/// The blocks a grid of large tiles needs to keep a GPU busy: about one per
/// SM of a mid-range Ada part, hand-picked from the sweep until tuning
/// measures each device.
const FULL_GRID: u64 = 32;

/// The `mma` of `target` that computes the product of `operands` that
/// `compute` sums: one whose multiplicands are both operands' type and
/// whose accumulator is the type the products are summed in.
pub(crate) fn instruction(
    compute: &ComputeDef,
    [lhs, rhs]: [TensorParamId; 2],
    target: &Target,
) -> Option<MmaInstruction> {
    let [output] = compute.outputs() else { return None };
    let Some(TensorExpr::Reduce { body, .. }) = output.expr.reduction() else { return None };
    let inputs = compute.inputs();
    let multiplicands = inputs[lhs].dtype;
    let accumulator = compute.dtype_of(body);
    let same_operands = inputs[rhs].dtype == multiplicands;
    let matches = |mma: &&MmaInstruction| {
        mma.multiplicands == multiplicands && mma.accumulator == accumulator
    };
    target.mma().iter().find(matches).copied().filter(|_| same_operands)
}

/// Tiles a matrix product's `loops` for `instruction` of `target`: the
/// batch loops and the block rows and columns fused onto `blockIdx.x`; the
/// block's warps on `threadIdx.y` and `threadIdx.z`, their lanes on
/// `threadIdx.x`; and the reduction split into staged, pipelined steps,
/// inside which each warp multiplies its tiles with the steps and tiles
/// unrolled and the innermost `mma` tile tensorized.
pub(crate) fn tiled_on_tensor_cores(
    schedule: &mut Schedule<Open>,
    loops: &GemmLoops,
    (instruction, target): (MmaInstruction, &Target),
) -> Result<(), ScheduleError> {
    let shape = instruction.shape;
    let tile = tile_for(schedule, loops, shape);
    let row_factors = [tile.warps.rows, tile.per_warp.rows, shape.m.get()];
    let [row_block, row_warp, row_tile, m] = parts(schedule, loops.row, &row_factors)?;
    let column_factors = [tile.warps.columns, tile.per_warp.columns, shape.n.get()];
    let [column_block, column_warp, column_tile, n] =
        parts(schedule, loops.column, &column_factors)?;
    let [step, step_depth, k] = parts(schedule, loops.depth, &[tile.steps, shape.k.get()])?;
    let mut order = loops.batch.clone();
    order.extend([row_block, column_block, row_warp, column_warp]);
    order.extend([step, step_depth, row_tile, column_tile, m, n, k]);
    schedule.apply(Primitive::Reorder { order })?;
    bind_blocks(schedule, &loops.batch, [row_block, column_block])?;
    bind(schedule, row_warp, ThreadAxis::ThreadY)?;
    bind(schedule, column_warp, ThreadAxis::ThreadZ)?;
    stage_operands(schedule, loops.operands, step)?;
    swizzle_tiles(schedule, loops.operands)?;
    pipeline_step(schedule, loops.operands, (step, tile.step(shape)), target)?;
    for loop_ in [step_depth, row_tile, column_tile] {
        schedule.apply(Primitive::Unroll { loop_ })?;
    }
    schedule.apply(Primitive::Tensorize { m, n, k, instruction })
}

/// The large tile where the product of `loops`, unsplit, has at least
/// `FULL_GRID` of them with `shape`'s `mma`, and the small one otherwise.
fn tile_for(schedule: &Schedule<Open>, loops: &GemmLoops, shape: MmaShape) -> TensorCoreTile {
    let extent = |id: LoopId| schedule.loops().loops()[id].extent.get();
    let batch: u64 = loops.batch.iter().map(|&id| extent(id)).product();
    let blocks = |extent: u64, warps: u64, per_warp: u64, mma: Dim| {
        extent.div_ceil(warps * per_warp * mma.get())
    };
    let rows = blocks(extent(loops.row), LARGE.warps.rows, LARGE.per_warp.rows, shape.m);
    let columns =
        blocks(extent(loops.column), LARGE.warps.columns, LARGE.per_warp.columns, shape.n);
    if batch * rows * columns >= FULL_GRID { LARGE } else { SMALL }
}

/// Both `operands`' staged tiles swizzled as CUTLASS's layouts are, so the
/// rows an `ldmatrix` reads fall in different banks: each whose rows have
/// a swizzle and that holds whole blocks of it.
fn swizzle_tiles(
    schedule: &mut Schedule<Open>,
    operands: [TensorParamId; 2],
) -> Result<(), ScheduleError> {
    for tensor in operands {
        let Some(extents) = schedule.staged_extents(tensor)? else { continue };
        let element = schedule.compute().inputs()[tensor].dtype.element_size().get();
        if let Some(swizzle) = Swizzle::for_tile(&extents, element) {
            schedule.apply(Primitive::Swizzle { tensor, swizzle })?;
        }
    }
    Ok(())
}
