//! Variadic elementwise ops and `Where`, every operand broadcast together.

use crate::error::{InterpError, Malformed};
use crate::extents::{Extents, broadcast_map};
use crate::movement::{join, take};
use crate::numeric::{Elements, count_to_f32, f64_to_f32, maximum, minimum, store, widen};
use ptxc_core::{HostTensor, TensorData};
use ptxc_graph::{TensorType, VariadicOp};

pub(in crate::eval) fn variadic(
    op: VariadicOp,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let extents = operands
        .iter()
        .map(|operand| Extents::of(operand.shape()))
        .collect::<Result<Vec<_>, _>>()?;
    let out = Extents::of(output.shape())?;
    let elements: Vec<Elements> = operands.iter().map(|operand| widen(operand.data())).collect();
    let combined = if let Some(floats) = collect(&elements, |e| match e {
        Elements::Float(values) => Some(values.as_slice()),
        _ => None,
    }) {
        let inputs: Vec<_> = floats.into_iter().zip(&extents).collect();
        Elements::Float(broadcast_map(&inputs, &out, |args| float_combine(op, args)))
    } else if let Some(ints) = collect(&elements, |e| match e {
        Elements::Int(values) => Some(values.as_slice()),
        _ => None,
    }) {
        let inputs: Vec<_> = ints.into_iter().zip(&extents).collect();
        let values: Option<Vec<i64>> =
            broadcast_map(&inputs, &out, |args| int_combine(op, args)).into_iter().collect();
        Elements::Int(values.ok_or_else(|| Malformed::DType(output.dtype()))?)
    } else {
        return Err(Malformed::DType(output.dtype()).into());
    };
    store(combined, output.dtype())
}

/// Every operand's values, if all are in the domain `pick` selects.
fn collect<'a, T>(
    elements: &'a [Elements],
    pick: impl Fn(&'a Elements) -> Option<&'a [T]>,
) -> Option<Vec<&'a [T]>> {
    elements.iter().map(pick).collect()
}

/// One output element. `Max` and `Min` propagate NaN, as NumPy's
/// `maximum` and `minimum`; `Mean` divides the f32 sum by the count.
fn float_combine(op: VariadicOp, args: &[f32]) -> f32 {
    let wide = || args.iter().map(|&v| f64::from(v));
    let sum = || args.iter().sum::<f32>();
    match op {
        VariadicOp::Max => f64_to_f32(wide().fold(f64::NEG_INFINITY, maximum)),
        VariadicOp::Min => f64_to_f32(wide().fold(f64::INFINITY, minimum)),
        VariadicOp::Sum => sum(),
        VariadicOp::Mean => sum() / count_to_f32(args.len()),
    }
}

/// One integer output element; only `Max` and `Min` take integers.
fn int_combine(op: VariadicOp, args: &[i64]) -> Option<i64> {
    match op {
        VariadicOp::Max => args.iter().copied().max(),
        VariadicOp::Min => args.iter().copied().min(),
        VariadicOp::Sum | VariadicOp::Mean => None,
    }
}

/// `Where(condition, x, y)`: `x`'s element where the condition holds, else
/// `y`'s, for any element type.
pub(in crate::eval) fn select(
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let [condition, x, y] = operands else { return Err(Malformed::Operands.into()) };
    let TensorData::Bool(flags) = condition.data() else {
        return Err(Malformed::DType(condition.dtype()).into());
    };
    let (c, xe, ye) =
        (Extents::of(condition.shape())?, Extents::of(x.shape())?, Extents::of(y.shape())?);
    let out = Extents::of(output.shape())?;
    let positions: Vec<usize> = out
        .indices()
        .map(|index| {
            if flags[c.broadcast_offset(&index)] {
                xe.broadcast_offset(&index)
            } else {
                xe.count() + ye.broadcast_offset(&index)
            }
        })
        .collect();
    Ok(take(&join(&[x.data(), y.data()])?, &positions))
}
