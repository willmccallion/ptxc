//! Indexing ops and their type rules.

use super::accepts::Accepts;
use super::{TypeError, check_axis, exact_operands, require_rank};
use crate::tensor_type::{Layout, TensorType};
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
