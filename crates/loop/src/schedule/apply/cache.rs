//! Applies `cache_read`: an input's tile staged in shared memory at a loop.

use crate::compute::{ComputeDef, IndexVar, Scalar, TensorExpr, TensorParamId};
use crate::expr::ExprId;
use crate::schedule::error::ScheduleError;
use crate::schedule::loop_tree::{CacheRead, LoopId, LoopTree};

/// Where a load sits in its output's expression.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::schedule) enum LoadSite {
    /// In a reduction's body, run at every point of the reduction loops.
    InReduction,
    /// Outside any reduction, run once per output point.
    OutsideReduction,
}

/// One load of an input: its compute-level index, its padding and where
/// it sits.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(in crate::schedule) struct Load {
    pub(in crate::schedule) index: Vec<ExprId>,
    pub(in crate::schedule) padding: Option<Scalar>,
    pub(in crate::schedule) site: LoadSite,
}

impl LoopTree {
    /// Stages `tensor` at serial loop `at`. Its loads must share one index
    /// that reads no gather, and one padding. Where `at` sits in the nest is checked by
    /// validation, since later primitives can move or replace it.
    pub(in crate::schedule) fn cache_read(
        &mut self,
        compute: &ComputeDef,
        tensor: TensorParamId,
        at: LoopId,
    ) -> Result<(), ScheduleError> {
        // NOTE: TVM's `cache_read` makes a stage that `compute_at` then
        // attaches to a loop, and bound inference unions every load's
        // region. ptxc takes the loop with the stage, as a tile has nowhere
        // else to be filled before `compute_at` exists, and requires one
        // shared index, so the tile is the box that index covers.
        let input = compute.inputs().get(tensor).ok_or(ScheduleError::UnknownTensor(tensor))?;
        let found = self.current_loop(at)?;
        if !found.annotation.runs_serially() {
            let (tensor, at) = (input.name.clone(), found.name.clone());
            return Err(ScheduleError::CacheNotSerial { tensor, at });
        }
        if self.caches.iter().any(|cache| cache.tensor == tensor) {
            return Err(ScheduleError::AlreadyCached(input.name.clone()));
        }
        let (index, padding) = shared_read(compute, tensor)?;
        self.caches.push(CacheRead { tensor, at, index, padding, swizzle: None });
        Ok(())
    }
}

/// The one index every load of `tensor` reads, and its padding.
fn shared_read(
    compute: &ComputeDef,
    tensor: TensorParamId,
) -> Result<(Vec<ExprId>, Option<Scalar>), ScheduleError> {
    let name = || compute.inputs()[tensor].name.clone();
    let loads = loads_of(compute, tensor);
    let Some(first) = loads.first() else {
        return Err(ScheduleError::CacheUnread(name()));
    };
    if loads.iter().any(|load| (&load.index, load.padding) != (&first.index, first.padding)) {
        return Err(ScheduleError::CacheLoads(name()));
    }
    let gathers = first.index.iter().flat_map(|&id| compute.exprs().vars(id));
    if gathers.into_iter().any(|var| matches!(compute.vars()[var], IndexVar::Loaded { .. })) {
        return Err(ScheduleError::CacheGathered(name()));
    }
    Ok((first.index.clone(), first.padding))
}

/// Every load of `tensor` in `compute`'s outputs.
pub(in crate::schedule) fn loads_of(compute: &ComputeDef, tensor: TensorParamId) -> Vec<Load> {
    let mut loads = Vec::new();
    for output in compute.outputs() {
        collect_loads(&output.expr, tensor, LoadSite::OutsideReduction, &mut loads);
    }
    loads
}

fn collect_loads(expr: &TensorExpr, tensor: TensorParamId, site: LoadSite, loads: &mut Vec<Load>) {
    let mut each = |operand: &TensorExpr| collect_loads(operand, tensor, site, loads);
    match expr {
        TensorExpr::Load { tensor: read, index, padding } => {
            if *read == tensor {
                loads.push(Load { index: index.clone(), padding: *padding, site });
            }
        }
        TensorExpr::Const(_) | TensorExpr::Index { .. } => {}
        TensorExpr::Unary(_, value) | TensorExpr::Cast { value, .. } => each(value),
        TensorExpr::Binary(_, lhs, rhs) => {
            each(lhs);
            each(rhs);
        }
        TensorExpr::Select { condition, then, otherwise } => {
            each(condition);
            each(then);
            each(otherwise);
        }
        TensorExpr::Reduce { body, .. } | TensorExpr::ArgReduce { body, .. } => {
            collect_loads(body, tensor, LoadSite::InReduction, loads);
        }
    }
}
