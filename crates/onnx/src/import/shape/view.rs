//! `Reshape` and the ops imported as it, `Transpose`, `Tile` and `Concat`.

use super::{input_dims, invalid, shape};
use crate::context::NodeContext;
use crate::error::{ImportError, ImportErrorKind};
use crate::lowered::Lowered;
use ptxc_core::{Dim, Rank, Shape};
use ptxc_graph::{Op, RepeatCount, TileAttrs, TransposeAttrs};

/// `Reshape`: `0` copies the input's dim at that position unless `allowzero`
/// (opset 14) makes it a real zero; one `-1` takes whatever is left.
pub(super) fn reshape(context: &mut NodeContext<'_>) -> Result<Shape, ImportError> {
    let requested = context.ints(1)?;
    let allow_zero = context.attrs.int_or("allowzero", 0)? != 0;
    let input = context.ty(0)?.shape().clone();
    let mut dims: Vec<Option<u64>> = Vec::with_capacity(requested.len());
    for (position, &entry) in requested.iter().enumerate() {
        dims.push(match entry {
            -1 => None,
            0 if !allow_zero => Some(
                input
                    .dims()
                    .get(position)
                    .ok_or_else(|| invalid("Reshape copies a dim the input lacks"))?
                    .get(),
            ),
            _ => Some(
                u64::try_from(entry)
                    .map_err(|_| invalid("Reshape has a negative dim other than -1"))?,
            ),
        });
    }
    let known: u64 = dims.iter().flatten().product();
    let total = input.element_count().get();
    let inferred = match dims.iter().filter(|dim| dim.is_none()).count() {
        0 => None,
        1 if known != 0 && total % known == 0 => Some(total / known),
        _ => return Err(invalid("Reshape cannot infer its -1")),
    };
    let resolved: Vec<Dim> = dims
        .into_iter()
        .map(|dim| dim.or(inferred).map(Dim::new))
        .collect::<Option<_>>()
        .ok_or_else(|| invalid("Reshape cannot infer its -1"))?;
    Shape::new(resolved).map_err(|error| invalid(error.to_string()))
}

/// `Squeeze`: removes the listed axes, or every axis of extent 1.
pub(super) fn squeeze(context: &NodeContext<'_>) -> Result<Shape, ImportError> {
    let dims = input_dims(context)?;
    let rank = Rank::new(dims.len());
    let axes: Vec<usize> = if context.has_input(1) {
        context
            .ints(1)?
            .iter()
            .map(|&axis| context.axis(axis, rank).map(ptxc_core::Axis::index))
            .collect::<Result<_, _>>()?
    } else {
        (0..dims.len()).filter(|&axis| dims[axis] == Dim::new(1)).collect()
    };
    if axes.iter().any(|&axis| dims[axis] != Dim::new(1)) {
        return Err(invalid("Squeeze removes an axis whose extent is not 1"));
    }
    let kept: Vec<Dim> = dims
        .iter()
        .enumerate()
        .filter(|(axis, _)| !axes.contains(axis))
        .map(|(_, &dim)| dim)
        .collect();
    Shape::new(kept).map_err(|error| invalid(error.to_string()))
}

/// `Unsqueeze`: inserts extent-1 axes at the listed positions of the output.
pub(super) fn unsqueeze(context: &NodeContext<'_>) -> Result<Shape, ImportError> {
    let dims = input_dims(context)?;
    let inserted = context.ints(1)?;
    let rank = Rank::new(dims.len() + inserted.len());
    let positions: Vec<usize> = inserted
        .iter()
        .map(|&axis| context.axis(axis, rank).map(ptxc_core::Axis::index))
        .collect::<Result<_, _>>()?;
    let mut remaining = dims.into_iter();
    let output: Vec<Dim> = (0..rank.get())
        .map(|axis| if positions.contains(&axis) { Some(Dim::new(1)) } else { remaining.next() })
        .collect::<Option<_>>()
        .ok_or_else(|| invalid("Unsqueeze axes repeat"))?;
    Shape::new(output).map_err(|error| invalid(error.to_string()))
}

/// `Flatten` at `axis` in `[-rank, rank]`: the dims before it and after it,
/// each multiplied together.
pub(super) fn flatten(context: &mut NodeContext<'_>) -> Result<Shape, ImportError> {
    let dims = input_dims(context)?;
    let written = context.attrs.int_or("axis", 1)?;
    let rank = i64::try_from(dims.len()).map_err(|_| invalid("Flatten input rank"))?;
    let resolved = if written < 0 { written + rank } else { written };
    let axis = usize::try_from(resolved)
        .ok()
        .filter(|&axis| axis <= dims.len())
        .ok_or_else(|| invalid("Flatten axis is out of range"))?;
    let product = |dims: &[Dim]| {
        dims.iter().try_fold(1_i64, |acc, dim| {
            i64::try_from(dim.get()).ok().and_then(|d| acc.checked_mul(d))
        })
    };
    let outer =
        product(&dims[..axis.min(dims.len())]).ok_or_else(|| invalid("Flatten overflows"))?;
    let inner =
        product(&dims[axis.min(dims.len())..]).ok_or_else(|| invalid("Flatten overflows"))?;
    shape(&[outer, inner])
}

/// `Transpose`, reversing the axes when `perm` is absent.
pub(super) fn transpose(context: &mut NodeContext<'_>) -> Result<Op, ImportError> {
    let rank = context.ty(0)?.shape().rank();
    let permutation = match context.attrs.ints("perm")? {
        Some(perm) => {
            perm.iter().map(|&axis| context.axis(axis, rank)).collect::<Result<_, _>>()?
        }
        None => (0..rank.get()).rev().map(ptxc_core::Axis::new).collect(),
    };
    Ok(Op::Transpose(TransposeAttrs { permutation }))
}

pub(super) fn tile(context: &NodeContext<'_>) -> Result<Op, ImportError> {
    let repeats = context
        .ints(1)?
        .iter()
        .map(|&count| u64::try_from(count).map(RepeatCount::new))
        .collect::<Result<_, _>>()
        .map_err(|_| invalid("Tile repeats must not be negative"))?;
    Ok(Op::Tile(TileAttrs { repeats }))
}

pub(super) fn concat(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let rank = context.ty(0)?.shape().rank();
    let axis = context.attrs.int("axis")?.ok_or(ImportErrorKind::Missing("Concat axis"))?;
    let axis = context.axis(axis, rank)?;
    Ok(Lowered::on_inputs(Op::Concat(axis), context.input_count()))
}
