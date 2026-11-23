//! Imports `Gather`, `GatherElements` and `GatherND`.

use crate::context::NodeContext;
use crate::error::{ImportError, ImportErrorKind};
use crate::lowered::Lowered;
use ptxc_core::{Axis, Rank};
use ptxc_graph::Op;

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    let make: fn(Axis) -> Op = match op {
        "Gather" => Op::Gather,
        "GatherElements" => Op::GatherElements,
        "GatherND" => return Some(gather_nd(context)),
        _ => return None,
    };
    Some(axis(context).map(|axis| Lowered::on_inputs(make(axis), 2)))
}

fn axis(context: &mut NodeContext<'_>) -> Result<Axis, ImportError> {
    let rank = context.ty(0)?.shape().rank();
    context.axis_attr("axis", 0, rank)
}

fn gather_nd(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let batch_dims = context.attrs.int_or("batch_dims", 0)?;
    let batch_dims = usize::try_from(batch_dims).map_err(|_| ImportErrorKind::Invalid {
        field: "batch_dims",
        problem: format!("GatherND batch_dims {batch_dims} is negative"),
    })?;
    Ok(Lowered::on_inputs(Op::GatherNd(Rank::new(batch_dims)), 2))
}
