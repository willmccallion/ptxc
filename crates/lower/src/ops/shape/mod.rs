//! Data movement: each op one stage whose output element reads the input
//! element its index arithmetic names. Reads that a select may discard are
//! clamped into the input, since both branches of a select are evaluated
//! (D40).

mod pad;
mod slice;
mod view;

pub(super) use pad::{pad, trilu};
pub(super) use slice::{concat, slice, split};
pub(super) use view::{expand, reshape, tile, transpose};

use super::NodeContext;
use crate::error::LowerError;
use ptxc_core::{Axis, DType, Dim};
use ptxc_loop::{ExprArena, ExprId, TensorExpr};

/// `index + delta`.
pub(in crate::ops) fn shifted(exprs: &mut ExprArena, index: ExprId, delta: i64) -> ExprId {
    if delta == 0 {
        return index;
    }
    let delta = exprs.constant(delta);
    exprs.add(index, delta)
}

/// `index` limited to `[0, last]`.
pub(in crate::ops) fn clamped(exprs: &mut ExprArena, index: ExprId, last: i64) -> ExprId {
    if last == 0 {
        return exprs.constant(0);
    }
    let (zero, last) = (exprs.constant(0), exprs.constant(last));
    let raised = exprs.max(index, zero);
    exprs.min(raised, last)
}

/// `spatial` with the index of `axis` replaced by `index`.
pub(in crate::ops) fn replaced(spatial: &[ExprId], axis: Axis, index: ExprId) -> Vec<ExprId> {
    spatial
        .iter()
        .enumerate()
        .map(|(position, &var)| if position == axis.index() { index } else { var })
        .collect()
}

/// An index expression's value, for comparing positions.
pub(in crate::ops) const fn position(index: ExprId) -> TensorExpr {
    TensorExpr::Index { index, dtype: DType::I64 }
}

/// The flat row-major position of `indices` in a tensor of `dims`.
fn linearized(
    cx: &NodeContext<'_>,
    exprs: &mut ExprArena,
    indices: &[ExprId],
    dims: &[Dim],
) -> Result<ExprId, LowerError> {
    let Some((&first, rest)) = indices.split_first() else { return Ok(exprs.constant(0)) };
    let mut flat = first;
    for (&index, &dim) in rest.iter().zip(dims.iter().skip(1)) {
        let extent = exprs.constant(cx.signed(dim)?);
        let scaled = exprs.mul(flat, extent);
        flat = exprs.add(scaled, index);
    }
    Ok(flat)
}

/// The row-major indices of flat position `flat` in a tensor of `dims`.
fn delinearized(
    cx: &NodeContext<'_>,
    exprs: &mut ExprArena,
    flat: ExprId,
    dims: &[Dim],
) -> Result<Vec<ExprId>, LowerError> {
    let mut stride = 1_i64;
    let mut indices = Vec::with_capacity(dims.len());
    for &dim in dims.iter().rev() {
        let extent = cx.signed(dim)?;
        let quotient = if stride == 1 {
            flat
        } else {
            let stride = exprs.constant(stride);
            exprs.floor_div(flat, stride)
        };
        let extent_expr = exprs.constant(extent);
        indices.push(exprs.floor_mod(quotient, extent_expr));
        stride = stride.saturating_mul(extent);
    }
    indices.reverse();
    Ok(indices)
}
