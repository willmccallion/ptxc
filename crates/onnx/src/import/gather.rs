//! Imports `Gather` and `GatherElements`.

use crate::context::NodeContext;
use crate::error::ImportError;
use crate::lowered::Lowered;
use ptxc_graph::Op;

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    let make: fn(ptxc_core::Axis) -> Op = match op {
        "Gather" => Op::Gather,
        "GatherElements" => Op::GatherElements,
        _ => return None,
    };
    Some(axis(context).map(|axis| Lowered::on_inputs(make(axis), 2)))
}

fn axis(context: &mut NodeContext<'_>) -> Result<ptxc_core::Axis, ImportError> {
    let rank = context.ty(0)?.shape().rank();
    context.axis_attr("axis", 0, rank)
}
