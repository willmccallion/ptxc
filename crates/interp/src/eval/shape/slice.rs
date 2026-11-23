//! `Slice` and `Split`.

use crate::error::{InterpError, Malformed};
use crate::eval::{first, result};
use crate::extents::Extents;
use crate::movement::take;
use crate::numeric::signed;
use ptxc_core::{HostTensor, TensorData};
use ptxc_graph::{SliceAttrs, SplitAttrs, TensorType};

/// `Slice`: along each sliced axis, output index `i` reads `start + step * i`.
pub(in crate::eval) fn slice(
    attrs: &SliceAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let (extents, out) = (Extents::of(input.shape())?, Extents::of(output.shape())?);
    let positions = out
        .indices()
        .map(|mut index| {
            for slice in &attrs.axes {
                let axis = slice.axis.index();
                let source =
                    i128::from(slice.start) + i128::from(slice.step.get()) * signed(index[axis]);
                index[axis] = usize::try_from(source).ok().ok_or(Malformed::OutOfBounds)?;
            }
            Ok(extents.offset(&index))
        })
        .collect::<Result<Vec<_>, InterpError>>()?;
    Ok(take(input.data(), &positions))
}

/// `Split`: consecutive pieces of the axis, one output each.
pub(in crate::eval) fn split(
    attrs: &SplitAttrs,
    operands: &[&HostTensor],
    outputs: &[&TensorType],
) -> Result<Vec<HostTensor>, InterpError> {
    let input = first(operands)?;
    let extents = Extents::of(input.shape())?;
    let axis = attrs.axis.index();
    let mut start = 0;
    let mut pieces = Vec::with_capacity(outputs.len());
    for ty in outputs {
        let out = Extents::of(ty.shape())?;
        let positions: Vec<usize> = out
            .indices()
            .map(|mut index| {
                index[axis] += start;
                extents.offset(&index)
            })
            .collect();
        start += out.dims()[axis];
        pieces.push(result(ty, take(input.data(), &positions))?);
    }
    Ok(pieces)
}
