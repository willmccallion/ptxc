//! Imports reductions and arg-reductions.

use crate::context::NodeContext;
use crate::error::ImportError;
use crate::lowered::Lowered;
use ptxc_graph::{ArgKind, ArgReduceAttrs, KeepDims, Op, ReduceAttrs, ReduceOp, TieBreak};

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    let reduce_op = match op {
        "ReduceSum" => ReduceOp::Sum,
        "ReduceMean" => ReduceOp::Mean,
        "ReduceProd" => ReduceOp::Prod,
        "ReduceMax" => ReduceOp::Max,
        "ReduceMin" => ReduceOp::Min,
        "ReduceSumSquare" => ReduceOp::SumSquare,
        "ReduceL1" => ReduceOp::L1,
        "ReduceL2" => ReduceOp::L2,
        "ReduceLogSum" => ReduceOp::LogSum,
        "ReduceLogSumExp" => ReduceOp::LogSumExp,
        "ArgMax" => return Some(arg_reduce(context, ArgKind::Max)),
        "ArgMin" => return Some(arg_reduce(context, ArgKind::Min)),
        _ => return None,
    };
    Some(reduce(context, reduce_op))
}

fn keep_dims(context: &mut NodeContext<'_>) -> Result<KeepDims, ImportError> {
    Ok(if context.attrs.int_or("keepdims", 1)? == 0 { KeepDims::Drop } else { KeepDims::Keep })
}

/// A reduction with explicit axes (D32). `axes` is an input from opset 13
/// for `ReduceSum` and from 18 for the rest, an attribute before. No axes
/// means every axis, unless `noop_with_empty_axes` makes the node an
/// identity.
fn reduce(context: &mut NodeContext<'_>, op: ReduceOp) -> Result<Lowered, ImportError> {
    let axes_are_input = context.opset.at_least(18) || op == ReduceOp::Sum;
    let written: Vec<i64> = if axes_are_input {
        if context.has_input(1) { context.ints(1)? } else { Vec::new() }
    } else {
        context.attrs.ints("axes")?.map(<[i64]>::to_vec).unwrap_or_default()
    };
    let keep_dims = keep_dims(context)?;
    let noop = context.attrs.int_or("noop_with_empty_axes", 0)? != 0;
    let rank = context.ty(0)?.shape().rank();
    if written.is_empty() && noop {
        return Ok(Lowered::Alias(0));
    }
    let axes = if written.is_empty() {
        rank.axes().collect()
    } else {
        written.iter().map(|&axis| context.axis(axis, rank)).collect::<Result<_, _>>()?
    };
    Ok(Lowered::on_inputs(Op::Reduce(ReduceAttrs { op, axes, keep_dims }), 1))
}

fn arg_reduce(context: &mut NodeContext<'_>, kind: ArgKind) -> Result<Lowered, ImportError> {
    let rank = context.ty(0)?.shape().rank();
    let axis = context.axis_attr("axis", 0, rank)?;
    let keep_dims = keep_dims(context)?;
    let tie = if context.attrs.int_or("select_last_index", 0)? == 0 {
        TieBreak::First
    } else {
        TieBreak::Last
    };
    Ok(Lowered::on_inputs(Op::ArgReduce(ArgReduceAttrs { kind, axis, keep_dims, tie }), 1))
}
