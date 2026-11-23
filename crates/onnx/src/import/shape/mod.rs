//! Imports reshape, transpose, concat, split, slice, squeeze, unsqueeze, expand,
//! flatten, tile, pad and trilu. Shape-like operands must be known at import
//! (D32); `Squeeze`, `Unsqueeze` and `Flatten` become `Reshape`.

mod pad;
mod slice;
mod view;

use crate::context::NodeContext;
use crate::error::{ImportError, ImportErrorKind};
use crate::lowered::Lowered;
use pad::{pad, trilu};
use ptxc_core::{Dim, Shape};
use ptxc_graph::Op;
use slice::{slice, split};
use view::{concat, flatten, reshape, squeeze, tile, transpose, unsqueeze};

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    let single = |result: Result<Op, ImportError>| result.map(|op| Lowered::on_inputs(op, 1));
    Some(match op {
        "Reshape" => single(reshape(context).map(Op::Reshape)),
        "Squeeze" => single(squeeze(context).map(Op::Reshape)),
        "Unsqueeze" => single(unsqueeze(context).map(Op::Reshape)),
        "Flatten" => single(flatten(context).map(Op::Reshape)),
        "Expand" => single(context.ints(1).and_then(|dims| shape(&dims)).map(Op::Expand)),
        "Transpose" => single(transpose(context)),
        "Tile" => single(tile(context)),
        "Concat" => concat(context),
        "Slice" => single(slice(context)),
        "Split" => single(split(context)),
        "Pad" => pad(context),
        "Trilu" => single(trilu(context)),
        _ => return None,
    })
}

pub(super) fn invalid(problem: impl Into<String>) -> ImportError {
    ImportErrorKind::Invalid { field: "shape operand", problem: problem.into() }.into()
}

pub(super) fn shape(dims: &[i64]) -> Result<Shape, ImportError> {
    let dims = dims
        .iter()
        .map(|&dim| u64::try_from(dim).map(Dim::new))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid("a negative dimension"))?;
    Shape::new(dims).map_err(|error| invalid(error.to_string()))
}

pub(super) fn input_dims(context: &NodeContext<'_>) -> Result<Vec<Dim>, ImportError> {
    Ok(context.ty(0)?.shape().dims().to_vec())
}
