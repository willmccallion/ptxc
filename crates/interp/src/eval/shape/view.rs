//! `Expand`, `Transpose`, `Tile` and `Concat`.

use crate::error::{InterpError, Malformed};
use crate::eval::first;
use crate::extents::Extents;
use crate::movement::{join, take};
use ptxc_core::{Axis, HostTensor, TensorData};
use ptxc_graph::{TensorType, TransposeAttrs};

/// The input positions each output element reads, by a map from output
/// multi-index to input multi-index.
fn positions(
    input: &HostTensor,
    output: &TensorType,
    source: impl Fn(&[usize], &Extents) -> Vec<usize>,
) -> Result<Vec<usize>, InterpError> {
    let (input, out) = (Extents::of(input.shape())?, Extents::of(output.shape())?);
    Ok(out.indices().map(|index| input.offset(&source(&index, &input))).collect())
}

/// `Expand`: the input broadcast to the output's shape.
pub(in crate::eval) fn expand(
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let (extents, out) = (Extents::of(input.shape())?, Extents::of(output.shape())?);
    let positions: Vec<usize> =
        out.indices().map(|index| extents.broadcast_offset(&index)).collect();
    Ok(take(input.data(), &positions))
}

/// `Transpose`: output axis `i` is input axis `permutation[i]`.
pub(in crate::eval) fn transpose(
    attrs: &TransposeAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let positions = positions(input, output, |out, _| {
        let mut source = vec![0; out.len()];
        for (&index, axis) in out.iter().zip(&attrs.permutation) {
            source[axis.index()] = index;
        }
        source
    })?;
    Ok(take(input.data(), &positions))
}

/// `Tile`: each output index wraps around its input axis.
pub(in crate::eval) fn tile(
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let positions = positions(input, output, |out, extents| {
        out.iter().zip(extents.dims()).map(|(i, d)| i % d).collect()
    })?;
    Ok(take(input.data(), &positions))
}

/// `Concat`: for each block of the axes before `axis`, every operand's
/// block in turn.
pub(in crate::eval) fn concat(
    axis: Axis,
    operands: &[&HostTensor],
) -> Result<TensorData, InterpError> {
    let extents = operands
        .iter()
        .map(|operand| Extents::of(operand.shape()))
        .collect::<Result<Vec<_>, _>>()?;
    let first = extents.first().ok_or(Malformed::Operands)?;
    let outer: usize = first.dims()[..axis.index()].iter().product();
    let inners: Vec<usize> =
        extents.iter().map(|e| e.dims()[axis.index()..].iter().product()).collect();
    let bases: Vec<usize> = extents
        .iter()
        .scan(0, |base, e| {
            let start = *base;
            *base += e.count();
            Some(start)
        })
        .collect();
    let mut positions = Vec::with_capacity(extents.iter().map(Extents::count).sum());
    for block in 0..outer {
        for (&base, &inner) in bases.iter().zip(&inners) {
            positions.extend(base + block * inner..base + (block + 1) * inner);
        }
    }
    let data: Vec<&TensorData> = operands.iter().map(|operand| operand.data()).collect();
    Ok(take(&join(&data)?, &positions))
}
