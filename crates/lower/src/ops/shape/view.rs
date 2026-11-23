//! `Reshape`, `Expand`, `Transpose` and `Tile`.

use super::{delinearized, linearized};
use crate::error::LowerError;
use crate::ops::NodeContext;
use crate::stage::{Binding, LoweredGroup};
use ptxc_core::Shape;
use ptxc_graph::TransposeAttrs;

/// `Reshape` as a copy through the flat position. A row-major reshape moves
/// nothing, so the memory plan can make it an alias (a `View` group); until
/// it does, it is a stage like any other.
pub(in crate::ops) fn reshape(
    cx: &NodeContext<'_>,
    target: &Shape,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    cx.one_stage(|stage| {
        let spatial = stage.spatial().to_vec();
        let input_dims = cx.graph.value(input).ty().shape().dims().to_vec();
        let flat = linearized(cx, stage.exprs(), &spatial, target.dims())?;
        let index = delinearized(cx, stage.exprs(), flat, &input_dims)?;
        Ok(stage.load(Binding::Value(input), index).expr)
    })
}

/// `Expand`: the input broadcast to the output.
pub(in crate::ops) fn expand(cx: &NodeContext<'_>) -> Result<LoweredGroup, LowerError> {
    cx.elementwise(|operands| {
        let [x] = cx.exact(operands)?;
        Ok(x.expr.clone())
    })
}

/// `Transpose`: output axis `i` reads input axis `permutation[i]`.
pub(in crate::ops) fn transpose(
    cx: &NodeContext<'_>,
    attrs: &TransposeAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    cx.one_stage(|stage| {
        let spatial = stage.spatial().to_vec();
        let mut index = spatial.clone();
        for (&output_axis_var, input_axis) in spatial.iter().zip(&attrs.permutation) {
            let slot = index.get_mut(input_axis.index()).ok_or(LowerError::Operands(cx.id))?;
            *slot = output_axis_var;
        }
        Ok(stage.load(Binding::Value(input), index).expr)
    })
}

/// `Tile`: each output axis reads its input axis modulo the input extent.
pub(in crate::ops) fn tile(cx: &NodeContext<'_>) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    cx.one_stage(|stage| {
        let spatial = stage.spatial().to_vec();
        let dims = cx.graph.value(input).ty().shape().dims().to_vec();
        let mut index = Vec::with_capacity(dims.len());
        for (&var, &dim) in spatial.iter().zip(&dims) {
            // An empty input makes the output empty, so its index is never read.
            let wrapped = if dim.get() == 0 {
                var
            } else {
                let extent = stage.exprs().constant(cx.signed(dim)?);
                stage.exprs().floor_mod(var, extent)
            };
            index.push(wrapped);
        }
        Ok(stage.load(Binding::Value(input), index).expr)
    })
}
