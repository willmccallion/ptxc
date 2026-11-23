//! `Pad` and `Trilu`.

use super::{clamped, position, shifted, signed};
use crate::compose::{binary, select};
use crate::error::LowerError;
use crate::ops::NodeContext;
use crate::stage::{Binding, LoweredGroup};
use ptxc_core::Dim;
use ptxc_graph::{AxisPadding, PadAttrs, PadMode, TriangularPart, TriluAttrs};
use ptxc_loop::{BinaryOp, ExprArena, ExprId, Scalar, TensorExpr};

/// `Pad`. Constant padding selects the pad value outside the input, reading
/// the input at a clamped index; edge, reflect and wrap padding read the
/// input at an index folded into it, axis by axis, as NumPy's `pad` does.
pub(in crate::ops) fn pad(
    cx: &NodeContext<'_>,
    attrs: &PadAttrs,
) -> Result<LoweredGroup, LowerError> {
    let operands = cx.operands();
    let [input, ..] = operands[..] else { return Err(LowerError::Operands(cx.id)) };
    let dims = cx.graph.value(input).ty().shape().dims().to_vec();
    cx.one_stage(|stage| {
        let spatial = stage.spatial().to_vec();
        if attrs.mode != PadMode::Constant {
            let mut index = Vec::with_capacity(dims.len());
            for ((&var, &dim), padding) in spatial.iter().zip(&dims).zip(&attrs.axes) {
                index.push(folded(cx, stage.exprs(), attrs.mode, (var, dim, padding))?);
            }
            return Ok(stage.load(Binding::Value(input), index).expr);
        }
        let [_, value] = cx.values()?;
        let fill = stage.broadcast(value).expr;
        if dims.iter().any(|dim| dim.get() == 0) {
            return Ok(fill);
        }
        let mut index = Vec::with_capacity(dims.len());
        let mut conditions = Vec::new();
        for ((&var, &dim), padding) in spatial.iter().zip(&dims).zip(&attrs.axes) {
            if padding.begin.get() == 0 && padding.end.get() == 0 {
                index.push(var);
                continue;
            }
            let (begin, extent) = (signed(cx, Dim::new(padding.begin.get()))?, signed(cx, dim)?);
            let local = shifted(stage.exprs(), var, -begin);
            index.push(clamped(stage.exprs(), local, extent - 1));
            conditions.push(within(var, begin, begin + extent));
        }
        let read = stage.load(Binding::Value(input), index).expr;
        let all = conditions.into_iter().reduce(|both, next| binary(BinaryOp::And, both, next));
        let Some(inside) = all else { return Ok(read) };
        Ok(select(inside, read, fill))
    })
}

/// Whether `index` lies in `[start, end)`.
fn within(index: ExprId, start: i64, end: i64) -> TensorExpr {
    let at = |bound| TensorExpr::Const(Scalar::i64(bound));
    let after_start = binary(BinaryOp::GreaterOrEqual, position(index), at(start));
    let before_end = binary(BinaryOp::Less, position(index), at(end));
    binary(BinaryOp::And, after_start, before_end)
}

/// The input index output index `var` reads along an axis of `dim`
/// elements padded by `padding`, in edge, reflect or wrap mode.
fn folded(
    cx: &NodeContext<'_>,
    exprs: &mut ExprArena,
    mode: PadMode,
    (var, dim, padding): (ExprId, Dim, &AxisPadding),
) -> Result<ExprId, LowerError> {
    if padding.begin.get() == 0 && padding.end.get() == 0 {
        return Ok(var);
    }
    let (begin, extent) = (signed(cx, Dim::new(padding.begin.get()))?, signed(cx, dim)?);
    let local = shifted(exprs, var, -begin);
    Ok(match mode {
        PadMode::Constant | PadMode::Edge => clamped(exprs, local, extent - 1),
        PadMode::Wrap => {
            let extent = exprs.constant(extent);
            exprs.floor_mod(local, extent)
        }
        PadMode::Reflect if extent == 1 => exprs.constant(0),
        PadMode::Reflect => reflected(exprs, local, extent),
    })
}

/// `local` mirrored into `[0, extent)` without repeating the edges: folded
/// into one period of `2 * (extent - 1)`, then reflected back from the far
/// edge.
fn reflected(exprs: &mut ExprArena, local: ExprId, extent: i64) -> ExprId {
    let period = 2 * (extent - 1);
    let period_expr = exprs.constant(period);
    let folded = exprs.floor_mod(local, period_expr);
    let minus_one = exprs.constant(-1);
    let negated = exprs.mul(folded, minus_one);
    let back = shifted(exprs, negated, period);
    let mirrored = exprs.min(folded, back);
    // The interval bound cannot see that min(f, period - f) < extent, so
    // this no-op clamp lets `finish` prove the read in range.
    let last = exprs.constant(extent - 1);
    exprs.min(mirrored, last)
}

/// `Trilu`: the input on the kept side of the diagonal, zero elsewhere,
/// comparing `column - row` with the diagonal's offset.
pub(in crate::ops) fn trilu(
    cx: &NodeContext<'_>,
    attrs: TriluAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    cx.one_stage(|stage| {
        let x = stage.broadcast(input);
        let [.., row, column] = stage.spatial()[..] else {
            return Err(LowerError::Operands(cx.id));
        };
        let exprs = stage.exprs();
        let minus_one = exprs.constant(-1);
        let negated_row = exprs.mul(row, minus_one);
        let offset = exprs.add(column, negated_row);
        let comparison = match attrs.part {
            TriangularPart::Upper => BinaryOp::GreaterOrEqual,
            TriangularPart::Lower => BinaryOp::LessOrEqual,
        };
        let diagonal = TensorExpr::Const(Scalar::i64(attrs.diagonal.get()));
        let kept = binary(comparison, position(offset), diagonal);
        Ok(select(kept, x.expr, TensorExpr::Const(Scalar::zero(x.dtype))))
    })
}
