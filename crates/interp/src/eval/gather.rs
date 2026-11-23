//! `Gather` and `GatherElements`, with negative indices counting from the
//! end of the indexed axis.

use crate::error::{InterpError, Malformed};
use crate::extents::Extents;
use crate::movement::take;
use crate::numeric::{ints, signed};
use ptxc_core::{Axis, HostTensor, TensorData};
use ptxc_graph::TensorType;

/// An index into an axis of `extent` elements; negative counts from the end.
fn resolve(index: i64, extent: usize) -> Result<usize, InterpError> {
    let resolved = if index < 0 { i128::from(index) + signed(extent) } else { i128::from(index) };
    usize::try_from(resolved)
        .ok()
        .filter(|&i| i < extent)
        .ok_or(InterpError::IndexOutOfRange { index, extent })
}

fn data_and_indices<'t>(
    operands: &[&'t HostTensor],
) -> Result<(&'t HostTensor, Vec<i64>, Extents), InterpError> {
    let [data, indices] = operands else { return Err(Malformed::Operands.into()) };
    Ok((data, ints(indices.data())?, Extents::of(indices.shape())?))
}

/// `Gather`: the output is `data[:axis] + indices.shape + data[axis + 1:]`,
/// each index selecting a slice of `data` along `axis`.
pub(super) fn gather(
    axis: Axis,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let (data, indices, index_extents) = data_and_indices(operands)?;
    let (extents, out) = (Extents::of(data.shape())?, Extents::of(output.shape())?);
    let axis = axis.index();
    let index_rank = index_extents.rank();
    let positions = out
        .indices()
        .map(|index| {
            let selector = index_extents.offset(&index[axis..axis + index_rank]);
            let mut source = index[..axis].to_vec();
            source.push(resolve(indices[selector], extents.dims()[axis])?);
            source.extend_from_slice(&index[axis + index_rank..]);
            Ok(extents.offset(&source))
        })
        .collect::<Result<Vec<_>, InterpError>>()?;
    Ok(take(data.data(), &positions))
}

/// `GatherElements`: the output has the indices' shape, and each element is
/// `data` at the same position except along `axis`, where the index says.
pub(super) fn gather_elements(
    axis: Axis,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let (data, indices, _) = data_and_indices(operands)?;
    let (extents, out) = (Extents::of(data.shape())?, Extents::of(output.shape())?);
    let axis = axis.index();
    let positions = out
        .indices()
        .enumerate()
        .map(|(flat, mut index)| {
            index[axis] = resolve(indices[flat], extents.dims()[axis])?;
            Ok(extents.offset(&index))
        })
        .collect::<Result<Vec<_>, InterpError>>()?;
    Ok(take(data.data(), &positions))
}
