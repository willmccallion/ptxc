//! `Pad` and `Trilu`.

use crate::error::{InterpError, Malformed};
use crate::eval::first;
use crate::extents::Extents;
use crate::movement::{join, take, zero};
use crate::numeric::signed;
use ptxc_core::{HostTensor, TensorData};
use ptxc_graph::{PadAttrs, PadMode, TensorType, TriangularPart, TriluAttrs};

/// The input index output position `position` reads along an axis of
/// `extent` elements with `begin` elements added before it; `None` where
/// a constant pad fills it.
fn source(mode: PadMode, position: usize, begin: usize, extent: usize) -> Option<usize> {
    let shifted = signed(position) - signed(begin);
    let n = signed(extent);
    let index = match mode {
        PadMode::Constant => return usize::try_from(shifted).ok().filter(|&i| i < extent),
        PadMode::Edge => shifted.clamp(0, n - 1),
        PadMode::Wrap => shifted.rem_euclid(n),
        PadMode::Reflect if extent == 1 => 0,
        PadMode::Reflect => {
            let period = 2 * (n - 1);
            let folded = shifted.rem_euclid(period);
            if folded < n { folded } else { period - folded }
        }
    };
    usize::try_from(index).ok()
}

/// `Pad`: constant padding reads the value operand; edge, reflect and wrap
/// read the input, axis by axis, as NumPy's `pad` does.
pub(in crate::eval) fn pad(
    attrs: &PadAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let (input, value) = match operands {
        [input] => (input, None),
        [input, value] => (input, Some(value)),
        _ => return Err(Malformed::Operands.into()),
    };
    let (extents, out) = (Extents::of(input.shape())?, Extents::of(output.shape())?);
    let begins = attrs
        .axes
        .iter()
        .map(|axis| {
            usize::try_from(axis.begin.get())
                .map_err(|_| InterpError::TooLarge { elements: axis.begin.get() })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fill = extents.count();
    let positions: Vec<usize> = out
        .indices()
        .map(|index| {
            let sources: Option<Vec<usize>> = index
                .iter()
                .zip(&begins)
                .zip(extents.dims())
                .map(|((&position, &begin), &extent)| source(attrs.mode, position, begin, extent))
                .collect();
            sources.map_or(fill, |sources| extents.offset(&sources))
        })
        .collect();
    let data = match value {
        Some(value) => join(&[input.data(), value.data()])?,
        None => input.data().clone(),
    };
    Ok(take(&data, &positions))
}

/// `Trilu`: the elements on the kept side of the diagonal, zero elsewhere.
pub(in crate::eval) fn trilu(
    attrs: &TriluAttrs,
    operands: &[&HostTensor],
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let extents = Extents::of(input.shape())?;
    let fill = extents.count();
    let rank = extents.rank();
    let offset = i128::from(attrs.diagonal.get());
    let positions: Vec<usize> = extents
        .indices()
        .enumerate()
        .map(|(flat, index)| {
            let (row, column) = (signed(index[rank - 2]), signed(index[rank - 1]));
            let kept = match attrs.part {
                TriangularPart::Upper => column - row >= offset,
                TriangularPart::Lower => column - row <= offset,
            };
            if kept { flat } else { fill }
        })
        .collect();
    Ok(take(&join(&[input.data(), &zero(input.dtype())])?, &positions))
}
