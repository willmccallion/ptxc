//! Indexing ops and their type rules.

use super::accepts::Accepts;
use super::{TypeError, check_axis, exact_operands, require_rank};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::Rank;
use ptxc_core::{Axis, Shape};

/// ONNX `Gather(data, indices)` along `axis`: the result's dims are data's
/// before `axis`, then the indices' shape, then data's after `axis`.
pub(super) fn infer_gather(
    axis: Axis,
    operands: &[&TensorType],
) -> Result<Vec<TensorType>, TypeError> {
    let [data, indices] = exact_operands(operands)?;
    Accepts::Index.check(indices.dtype())?;
    check_axis(data.shape(), axis)?;
    let (before, rest) = data.shape().dims().split_at(axis.index());
    let after = &rest[1..];
    let dims = before.iter().chain(indices.shape().dims()).chain(after).copied().collect();
    Ok(vec![TensorType::new(data.dtype(), Shape::new(dims)?, Layout::RowMajor)])
}

/// ONNX `GatherElements(data, indices)` along `axis`: the result has the
/// indices' shape. Each element reads data at the same position on every
/// other axis, so the indices may not be longer than data there.
pub(super) fn infer_gather_elements(
    axis: Axis,
    operands: &[&TensorType],
) -> Result<Vec<TensorType>, TypeError> {
    let [data, indices] = exact_operands(operands)?;
    Accepts::Index.check(indices.dtype())?;
    check_axis(data.shape(), axis)?;
    require_rank(indices.shape(), data.shape().rank())?;
    let pairs = data.shape().dims().iter().zip(indices.shape().dims());
    for (other_axis, (&expected, &actual)) in data.shape().rank().axes().zip(pairs) {
        if other_axis != axis && actual > expected {
            return Err(TypeError::DimMismatch { axis: other_axis, expected, actual });
        }
    }
    Ok(vec![TensorType::new(data.dtype(), indices.shape().clone(), Layout::RowMajor)])
}

/// ONNX `GatherND(data, indices)` after `batch_dims` leading axes, on which
/// data and indices agree: each tuple along the indices' last axis selects
/// a slice of data, so the result's dims are the indices' but the last,
/// then data's after the batch axes and the tuple.
pub(super) fn infer_gather_nd(
    batch_dims: Rank,
    operands: &[&TensorType],
) -> Result<Vec<TensorType>, TypeError> {
    let [data, indices] = exact_operands(operands)?;
    Accepts::Index.check(indices.dtype())?;
    let batch = batch_dims.get();
    let minimum = Rank::new(batch + 1);
    for operand in [data, indices] {
        if operand.shape().rank() < minimum {
            return Err(TypeError::RankBelow { minimum, actual: operand.shape().rank() });
        }
    }
    let Some((length, index_outer)) = indices.shape().dims().split_last() else {
        return Err(TypeError::RankBelow { minimum, actual: indices.shape().rank() });
    };
    let sliced = Rank::new(data.shape().rank().get() - batch);
    let tuple =
        usize::try_from(length.get()).ok().filter(|&tuple| tuple >= 1 && tuple <= sliced.get());
    let Some(tuple) = tuple else {
        return Err(TypeError::IndexTupleLength { length: *length, axes: sliced });
    };
    let batch_pairs = data.shape().dims().iter().zip(index_outer).take(batch);
    for (axis, (&expected, &actual)) in data.shape().rank().axes().zip(batch_pairs) {
        if actual != expected {
            return Err(TypeError::DimMismatch { axis, expected, actual });
        }
    }
    let dims = index_outer.iter().chain(&data.shape().dims()[batch + tuple..]).copied().collect();
    Ok(vec![TensorType::new(data.dtype(), Shape::new(dims)?, Layout::RowMajor)])
}
