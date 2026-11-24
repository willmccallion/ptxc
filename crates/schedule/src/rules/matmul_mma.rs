//! Matrix multiplication on tensor cores, tiled as CUTLASS's tensor-op
//! GEMM is: each block computes a tile of the output, each warp a grid of
//! `mma` tiles of it held in fragments, and every step of the reduction
//! stages the block's slices of both operands in shared memory, from which
//! the warps load their fragments (D58).

use super::bind;
use super::matmul_loops::{batch_loops, bind_blocks, parts, stage_operands};
use super::matmul_shape::Product;
use crate::tiling::{TensorCoreTile, TileShape};
use ptxc_loop::{ComputeDef, Open, Primitive, Schedule, ScheduleError, TensorExpr, ThreadAxis};
use ptxc_target::{MmaInstruction, Target};

/// Hand-picked until the search tunes it: four warps, each holding 2 by 4
/// `mma` tiles, over a block tile of 64 by 64 at f16's m16n8, staged two
/// `mma` steps deep, so an f16 step stages 8 KiB.
const TILE: TensorCoreTile = TensorCoreTile {
    warps: TileShape { rows: 2, columns: 2 },
    per_warp: TileShape { rows: 2, columns: 4 },
    steps: 2,
};

/// The `mma` of `target` that computes `product` of `compute`: one whose
/// multiplicands are both operands' type and whose accumulator is the
/// type the products are summed in.
pub(crate) fn instruction(
    compute: &ComputeDef,
    product: Product,
    target: &Target,
) -> Option<MmaInstruction> {
    let [output] = compute.outputs() else { return None };
    let Some(TensorExpr::Reduce { body, .. }) = output.expr.reduction() else { return None };
    let inputs = compute.inputs();
    let multiplicands = inputs[product.lhs].dtype;
    let accumulator = compute.dtype_of(body);
    let same_operands = inputs[product.rhs].dtype == multiplicands;
    let matches = |mma: &&MmaInstruction| {
        mma.multiplicands == multiplicands && mma.accumulator == accumulator
    };
    target.mma().iter().find(matches).copied().filter(|_| same_operands)
}

/// Tiles `product`'s nest for `instruction`: the batch axes and the block
/// rows and columns fused onto `blockIdx.x`; the block's warps on
/// `threadIdx.y` and `threadIdx.z`, their lanes on `threadIdx.x`; and the
/// reduction split into staged steps, inside which each warp multiplies
/// its tiles with the steps and tiles unrolled and the innermost `mma`
/// tile tensorized.
pub(crate) fn tiled_on_tensor_cores(
    schedule: &mut Schedule<Open>,
    product: Product,
    instruction: MmaInstruction,
) -> Result<(), ScheduleError> {
    let shape = instruction.shape;
    let batch = batch_loops(schedule, product);
    let row_factors = [TILE.warps.rows, TILE.per_warp.rows, shape.m.get()];
    let [row_block, row_warp, row_tile, m] = parts(schedule, product.row, &row_factors)?;
    let column_factors = [TILE.warps.columns, TILE.per_warp.columns, shape.n.get()];
    let [column_block, column_warp, column_tile, n] =
        parts(schedule, product.column, &column_factors)?;
    let [step, step_depth, k] = parts(schedule, product.depth, &[TILE.steps, shape.k.get()])?;
    let mut order = batch.clone();
    order.extend([row_block, column_block, row_warp, column_warp]);
    order.extend([step, step_depth, row_tile, column_tile, m, n, k]);
    schedule.apply(Primitive::Reorder { order })?;
    bind_blocks(schedule, &batch, [row_block, column_block])?;
    bind(schedule, row_warp, ThreadAxis::ThreadY)?;
    bind(schedule, column_warp, ThreadAxis::ThreadZ)?;
    stage_operands(schedule, product, step)?;
    for loop_ in [step_depth, row_tile, column_tile] {
        schedule.apply(Primitive::Unroll { loop_ })?;
    }
    schedule.apply(Primitive::Tensorize { m, n, k, instruction })
}
