//! Imports max, average and global pooling.

use super::conv::read_window;
use crate::context::NodeContext;
use crate::error::{ImportError, ImportErrorKind};
use crate::lowered::Lowered;
use ptxc_core::Dim;
use ptxc_graph::{GlobalPoolKind, Op, PadCounting, PoolAttrs, PoolAxis, PoolKind, Rounding};

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    Some(match op {
        "MaxPool" => pool(context, PoolKind::Max),
        "AveragePool" => context.attrs.int_or("count_include_pad", 0).and_then(|include| {
            let counting = if include == 0 { PadCounting::Exclude } else { PadCounting::Include };
            pool(context, PoolKind::Average(counting))
        }),
        "GlobalMaxPool" => Ok(Lowered::on_inputs(Op::GlobalPool(GlobalPoolKind::Max), 1)),
        "GlobalAveragePool" => Ok(Lowered::on_inputs(Op::GlobalPool(GlobalPoolKind::Average), 1)),
        _ => return None,
    })
}

/// `MaxPool` or `AveragePool`. `MaxPool`'s indices output, which
/// `storage_order` describes, is not supported.
fn pool(context: &mut NodeContext<'_>, kind: PoolKind) -> Result<Lowered, ImportError> {
    if context.wants_output(1) {
        return Err(context.attrs.unsupported("Indices"));
    }
    if kind == PoolKind::Max && context.attrs.int_or("storage_order", 0)? != 0 {
        return Err(context.attrs.unsupported("storage_order"));
    }
    let kernel: Vec<Dim> = context
        .attrs
        .ints("kernel_shape")?
        .ok_or(ImportErrorKind::Missing("kernel_shape"))?
        .iter()
        .map(|&extent| u64::try_from(extent).map(Dim::new))
        .collect::<Result<_, _>>()
        .map_err(|_| ImportErrorKind::Invalid {
            field: "kernel_shape",
            problem: "negative".to_owned(),
        })?;
    let input = context.ty(0)?.shape().dims().to_vec();
    let windows = read_window(context, input.get(2..).unwrap_or_default(), &kernel)?;
    let rounding =
        if context.attrs.int_or("ceil_mode", 0)? == 0 { Rounding::Floor } else { Rounding::Ceil };
    let axes = kernel
        .into_iter()
        .zip(windows)
        .map(|(kernel, window)| PoolAxis { kernel, window })
        .collect();
    Ok(Lowered::on_inputs(Op::Pool(PoolAttrs { kind, axes, rounding }), 1))
}
