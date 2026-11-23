//! `Gather`, `GatherElements` and `GatherND`: reads of data at positions
//! held in an index tensor, through loop IR's loaded index variables (D4).

use super::NodeContext;
use super::shape::replaced;
use crate::builder::StageBuilder;
use crate::error::LowerError;
use crate::stage::{Binding, LoweredGroup};
use ptxc_core::{Axis, Dim, Rank};
use ptxc_graph::ValueId;
use ptxc_loop::{ExprId, Scalar, TensorExpr};

/// The position along an axis of `extent` that the index `loaded` names;
/// a negative index counts from the end, as ONNX's does.
fn resolved(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    loaded: ExprId,
    extent: Dim,
) -> Result<ExprId, LowerError> {
    // NOTE: ONNX makes an index outside [-extent, extent) an error, which a
    // kernel cannot raise; ptxc wraps it into the axis (D45), which also
    // keeps every read in bounds.
    let extent = stage.exprs().constant(cx.signed(extent)?);
    Ok(stage.exprs().floor_mod(loaded, extent))
}

/// The dims of `value`.
fn dims(cx: &NodeContext<'_>, value: ValueId) -> Vec<Dim> {
    cx.graph.value(value).ty().shape().dims().to_vec()
}

/// What an output element reading an empty axis holds: there are no valid
/// indices into it, so nothing is read.
fn nothing_to_read(cx: &NodeContext<'_>, data: ValueId) -> TensorExpr {
    TensorExpr::Const(Scalar::zero(cx.graph.value(data).ty().dtype()))
}

/// `Gather`: the output's axes are data's before `axis`, the indices',
/// then data's after `axis`; each index selects a slice of data.
pub(super) fn gather(cx: &NodeContext<'_>, axis: Axis) -> Result<LoweredGroup, LowerError> {
    let [data, indices] = cx.values()?;
    let extent = *dims(cx, data).get(axis.index()).ok_or(LowerError::Operands(cx.id))?;
    let index_rank = dims(cx, indices).len();
    cx.one_stage(|stage| {
        if extent.get() == 0 {
            return Ok(nothing_to_read(cx, data));
        }
        let spatial = stage.spatial().to_vec();
        let start = axis.index();
        let (before, rest) = spatial.split_at(start);
        let (selector, after) =
            rest.split_at_checked(index_rank).ok_or(LowerError::Operands(cx.id))?;
        let loaded = stage.loaded("j", Binding::Value(indices), selector.to_vec());
        let position = resolved(cx, stage, loaded, extent)?;
        let index = [before, &[position], after].concat();
        Ok(stage.load(Binding::Value(data), index).expr)
    })
}

/// `GatherElements`: the output has the indices' shape, and each element
/// reads data at its own position except along `axis`, where its index
/// says.
pub(super) fn gather_elements(
    cx: &NodeContext<'_>,
    axis: Axis,
) -> Result<LoweredGroup, LowerError> {
    let [data, indices] = cx.values()?;
    let extent = *dims(cx, data).get(axis.index()).ok_or(LowerError::Operands(cx.id))?;
    cx.one_stage(|stage| {
        if extent.get() == 0 {
            return Ok(nothing_to_read(cx, data));
        }
        let spatial = stage.spatial().to_vec();
        let loaded = stage.loaded("j", Binding::Value(indices), spatial.clone());
        let position = resolved(cx, stage, loaded, extent)?;
        Ok(stage.load(Binding::Value(data), replaced(&spatial, axis, position)).expr)
    })
}

/// `GatherND` after `batch` leading axes: each tuple along the indices'
/// last axis selects a slice of data within its batch.
pub(super) fn gather_nd(cx: &NodeContext<'_>, batch: Rank) -> Result<LoweredGroup, LowerError> {
    let [data, indices] = cx.values()?;
    let data_dims = dims(cx, data);
    let index_dims = dims(cx, indices);
    let (&length, outer) = index_dims.split_last().ok_or(LowerError::Operands(cx.id))?;
    let tuple_length = usize::try_from(length.get()).map_err(|_| LowerError::Operands(cx.id))?;
    let batch = batch.get();
    let tuple_axes =
        data_dims.get(batch..batch + tuple_length).ok_or(LowerError::Operands(cx.id))?.to_vec();
    cx.one_stage(|stage| {
        if tuple_axes.iter().any(|extent| extent.get() == 0) {
            return Ok(nothing_to_read(cx, data));
        }
        let spatial = stage.spatial().to_vec();
        let (outer_index, sliced) = spatial.split_at(outer.len());
        let mut index = outer_index.get(..batch).ok_or(LowerError::Operands(cx.id))?.to_vec();
        for (element, &extent) in (0_i64..).zip(&tuple_axes) {
            let at = stage.constant(element);
            let selector = [outer_index, &[at]].concat();
            let loaded = stage.loaded(&format!("j{element}"), Binding::Value(indices), selector);
            index.push(resolved(cx, stage, loaded, extent)?);
        }
        index.extend_from_slice(sliced);
        Ok(stage.load(Binding::Value(data), index).expr)
    })
}
