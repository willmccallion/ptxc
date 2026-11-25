//! Checks a tensorized nest against its definition: the tile's loops are
//! the innermost, inside a reduction step; the definition sums products of
//! two loads of the instruction's types; and both operands are staged in
//! shared memory outside the tile, each holding its tile as rows a fixed
//! distance apart with adjacent elements along rows or columns.

use super::error::ScheduleError;
use super::loop_tree::{CacheId, LoopAnnotation, LoopId, LoopKind, LoopTree, Tensorized};
use crate::compute::{ComputeDef, IndexVar, ReduceOp, TensorExpr, TensorParamId};
use crate::expr::{ExprId, VarId};
use crate::lower::operand_layout;
use crate::stmt::TileOrder;
use ptxc_core::{Dim, Idx};
use ptxc_target::ThreadsPerBlock;

/// A tensorized product as validation found it.
#[derive(Clone, Debug)]
pub(crate) struct TensorizedProduct {
    /// The tile and its instruction.
    pub(crate) tile: Tensorized,
    /// The operand whose rows are the tile's: `m × k`.
    pub(crate) lhs: Operand,
    /// The operand whose columns are the tile's: `k × n`.
    pub(crate) rhs: Operand,
}

/// One operand of a tensorized product, and how its staged tile steps.
#[derive(Clone, Debug)]
pub(crate) struct Operand {
    /// The stage of the input it reads.
    pub(crate) cache: CacheId,
    /// How the staged tile holds the operand's rows and columns.
    pub(crate) order: TileOrder,
    /// The staged tile's distance between the operand's rows, as stored.
    pub(crate) stride: Dim,
}

/// A load of the product's body, before its layout is known.
struct Load {
    tensor: TensorParamId,
    index: Vec<ExprId>,
}

/// The tensorized product of `tree`, if it is tensorized, checked against
/// `compute` with the staged tiles of `extents` filled by `threads`.
pub(super) fn check_tensorized(
    compute: &ComputeDef,
    tree: &LoopTree,
    (extents, threads): (&[Vec<Dim>], ThreadsPerBlock),
) -> Result<Option<TensorizedProduct>, ScheduleError> {
    let Some(tile) = tree.tensorized() else { return Ok(None) };
    check_place(tree, tile.loops)?;
    let [first, second] = loads(compute, tile)?;
    let [m, n, k] = tile.loops.map(|id| VarId::new(id.index()));
    let (lhs, rhs) =
        if reads(compute, tree, &first.index, m) { (first, second) } else { (second, first) };
    let layout = |load: Load, vars: [VarId; 3]| {
        let cache = staged(compute, tree, load.tensor, tile.loops[0])?;
        let found = operand_layout(compute, tree, (extents, threads), cache, vars);
        let name = || ScheduleError::TensorizeIndex(compute.inputs()[load.tensor].name.clone());
        let (order, stride) = found.ok_or_else(name)?;
        Ok(Operand { cache, order, stride })
    };
    let lhs = layout(lhs, [m, k, n])?;
    let rhs = layout(rhs, [k, n, m])?;
    Ok(Some(TensorizedProduct { tile, lhs, rhs }))
}

/// The tile's loops are the innermost serial loops, rows, columns, depth,
/// inside a serial reduction loop, with no reduction bound to threads, and
/// every serial reduction loop is outside the loops over the warp's tiles.
fn check_place(tree: &LoopTree, loops: [LoopId; 3]) -> Result<(), ScheduleError> {
    let bound = |id: LoopId| matches!(tree.loops()[id].annotation, LoopAnnotation::Bound(_));
    let serial: Vec<LoopId> = tree.nest().iter().copied().filter(|&id| !bound(id)).collect();
    let innermost = serial.len().checked_sub(3).map(|start| &serial[start..]);
    let in_place = innermost == Some(&loops[..])
        && loops.iter().all(|&id| tree.loops()[id].annotation == LoopAnnotation::Serial);
    if !in_place {
        return Err(ScheduleError::TensorizeNotInnermost);
    }
    let reduce = |id: &LoopId| tree.loops()[*id].kind == LoopKind::Reduce;
    let outside = &serial[..serial.len() - 3];
    let shared = tree.nest().iter().any(|id| reduce(id) && bound(*id));
    let Some(first) = outside.iter().position(reduce).filter(|_| !shared) else {
        return Err(ScheduleError::TensorizeNoStep);
    };
    let step_after_tile =
        outside[first..].windows(2).any(|pair| !reduce(&pair[0]) && reduce(&pair[1]));
    if step_after_tile { Err(ScheduleError::TensorizeStepInsideTiles) } else { Ok(()) }
}

/// The two loads the product multiplies: its one output a sum over one
/// axis of their product, each widened, if at all, to the accumulator's
/// type, and of the instruction's multiplicand type.
fn loads(compute: &ComputeDef, tile: Tensorized) -> Result<[Load; 2], ScheduleError> {
    let instruction = tile.instruction;
    let not_product = || ScheduleError::TensorizeNotProduct;
    let [output] = compute.outputs() else { return Err(not_product()) };
    let Some(TensorExpr::Reduce { op: ReduceOp::Sum, axes, body }) = output.expr.reduction() else {
        return Err(not_product());
    };
    let TensorExpr::Binary(crate::compute::BinaryOp::Mul, first, second) = body.as_ref() else {
        return Err(not_product());
    };
    if axes.len() != 1 || compute.dtype_of(body) != instruction.accumulator {
        return Err(not_product());
    }
    let load = |operand: &TensorExpr| -> Option<Load> {
        // A cast here widens to the accumulator's type, as the product is.
        let read = match operand {
            TensorExpr::Cast { value, .. } => value.as_ref(),
            other => other,
        };
        // A padded load's tile holds its padding, which the mma multiplies.
        let TensorExpr::Load { tensor, index, .. } = read else { return None };
        let dtype = compute.inputs()[*tensor].dtype;
        (dtype == instruction.multiplicands).then(|| Load { tensor: *tensor, index: index.clone() })
    };
    Ok([load(first).ok_or_else(not_product)?, load(second).ok_or_else(not_product)?])
}

/// Whether `index` reads `var` through any of its parts' axis indices.
fn reads(compute: &ComputeDef, tree: &LoopTree, index: &[ExprId], var: VarId) -> bool {
    index.iter().any(|&part| {
        compute.exprs().vars(part).into_iter().any(|read| match &compute.vars()[read] {
            IndexVar::Iter(axis) => tree.exprs().vars(tree.axis_index(*axis)).contains(&var),
            IndexVar::Loaded { .. } => false,
        })
    })
}

/// The stage of `tensor`, at a loop outside the tile, whose first loop is
/// `first`.
fn staged(
    compute: &ComputeDef,
    tree: &LoopTree,
    tensor: TensorParamId,
    first: LoopId,
) -> Result<CacheId, ScheduleError> {
    let position = |id: LoopId| tree.nest().iter().position(|&nested| nested == id);
    let outside =
        |at: LoopId| position(at).zip(position(first)).is_some_and(|(at, tile)| at < tile);
    tree.caches()
        .iter()
        .position(|cache| cache.tensor == tensor && outside(cache.at))
        .map(CacheId::new)
        .ok_or_else(|| ScheduleError::TensorizeUnstaged(compute.inputs()[tensor].name.clone()))
}
