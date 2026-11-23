//! `Gather`, `GatherElements` and `GatherND`, with negative indices counting
//! from the end of the indexed axis.

use crate::error::{InterpError, Malformed};
use crate::extents::Extents;
use crate::movement::take;
use crate::numeric::{ints, signed};
use ptxc_core::{Axis, HostTensor, Rank, TensorData};
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

/// `GatherND`: each tuple along the indices' last axis, after the batch
/// axes the indices' position gives, selects a contiguous slice of `data`.
pub(super) fn gather_nd(
    batch_dims: Rank,
    operands: &[&HostTensor],
) -> Result<TensorData, InterpError> {
    let (data, indices, index_extents) = data_and_indices(operands)?;
    let extents = Extents::of(data.shape())?;
    let batch = batch_dims.get();
    let (&length, outer) = index_extents.dims().split_last().ok_or(Malformed::Operands)?;
    if length == 0 {
        return Err(Malformed::Operands.into());
    }
    let tuple_axes = extents.dims().get(batch..batch + length).ok_or(Malformed::Operands)?;
    let slice: usize = extents.dims()[batch + length..].iter().product();
    let outer = Extents::from_dims(outer.to_vec());
    let mut positions = Vec::with_capacity(outer.count() * slice);
    for (row, tuple) in indices.chunks_exact(length).enumerate() {
        let mut source = outer.index_of(row)[..batch].to_vec();
        for (&index, &extent) in tuple.iter().zip(tuple_axes) {
            source.push(resolve(index, extent)?);
        }
        source.resize(extents.rank(), 0);
        let start = extents.offset(&source);
        positions.extend(start..start + slice);
    }
    Ok(take(data.data(), &positions))
}
