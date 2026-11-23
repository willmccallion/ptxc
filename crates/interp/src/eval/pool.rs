//! Max, average and global pooling.

use super::round_to_output;
use super::window::Windows;
use crate::error::{InterpError, Malformed};
use crate::eval::first;
use crate::extents::Extents;
use crate::numeric::{Elements, count_to_f64, int_range, maximum, store, wide_floats, widen};
use ptxc_core::{HostTensor, TensorData};
use ptxc_graph::{GlobalPoolKind, PadCounting, PoolAttrs, PoolKind, TensorType, WindowAxis};

/// Pooling windows over `[N, C, s…]`, and every window's taps.
struct Pooling {
    input: Extents,
    output: Extents,
    windows: Windows,
    taps: Vec<Vec<usize>>,
}

impl Pooling {
    fn of(attrs: &PoolAttrs, input: &HostTensor, output: &TensorType) -> Result<Self, InterpError> {
        let input = Extents::of(input.shape())?;
        let kernel: Vec<usize> = attrs
            .axes
            .iter()
            .map(|axis| {
                usize::try_from(axis.kernel.get())
                    .map_err(|_| InterpError::TooLarge { elements: axis.kernel.get() })
            })
            .collect::<Result<_, _>>()?;
        let kernel = Extents::from_dims(kernel);
        let taps = kernel.indices().collect();
        let windows: Vec<WindowAxis> = attrs.axes.iter().map(|axis| axis.window).collect();
        let windows = Windows::of(&windows, &input)?;
        Ok(Self { output: Extents::of(output.shape())?, input, windows, taps })
    }

    /// For each output element, the input offsets its window reads and how
    /// many of its taps lie within the input or its explicit padding.
    fn windows(&self) -> impl Iterator<Item = (Vec<usize>, usize)> + '_ {
        let strides = self.input.strides();
        self.output.indices().map(move |out| {
            let base = out[0] * strides[0] + out[1] * strides[1];
            let spatial = &out[2..];
            let sources =
                self.taps.iter().filter_map(|tap| self.windows.source(spatial, tap, &strides[2..]));
            let sources: Vec<usize> = sources.map(|source| base + source).collect();
            let padded =
                self.taps.iter().filter(|tap| self.windows.within_padding(spatial, tap)).count();
            (sources, padded)
        })
    }
}

pub(super) fn pool(
    attrs: &PoolAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let pooling = Pooling::of(attrs, input, output)?;
    match attrs.kind {
        PoolKind::Max => max_pool(&pooling, input, output),
        PoolKind::Average(counting) => {
            let values = wide_floats(input.data())?;
            let means: Vec<f64> = pooling
                .windows()
                .map(|(sources, padded)| {
                    let sum: f64 = sources.iter().map(|&i| values[i]).sum();
                    let count = match counting {
                        PadCounting::Exclude => sources.len(),
                        PadCounting::Include => padded,
                    };
                    sum / count_to_f64(count)
                })
                .collect();
            round_to_output(&means, output)
        }
    }
}

/// The largest element of each window; NaN wins, padding never does.
fn max_pool(
    pooling: &Pooling,
    input: &HostTensor,
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    match widen(input.data()) {
        Elements::Float(values) => {
            let maxima: Vec<f64> = pooling
                .windows()
                .map(|(sources, _)| {
                    sources.iter().map(|&i| f64::from(values[i])).fold(f64::NEG_INFINITY, maximum)
                })
                .collect();
            round_to_output(&maxima, output)
        }
        Elements::Int(values) => {
            let (lowest, _) = int_range(output.dtype())?;
            let maxima = pooling
                .windows()
                .map(|(sources, _)| sources.iter().map(|&i| values[i]).max().unwrap_or(lowest));
            store(Elements::Int(maxima.collect()), output.dtype())
        }
        Elements::Bool(_) => Err(Malformed::DType(input.dtype()).into()),
    }
}

/// The max or mean of each `[n, c]` plane over every spatial axis.
pub(super) fn global_pool(
    kind: GlobalPoolKind,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let values = wide_floats(input.data())?;
    let extents = Extents::of(input.shape())?;
    let planes = extents.dims().iter().take(2).product::<usize>();
    let plane = extents.dims().iter().skip(2).product::<usize>();
    let pooled: Vec<f64> = (0..planes)
        .map(|p| {
            let elements = &values[p * plane..(p + 1) * plane];
            match kind {
                GlobalPoolKind::Max => elements.iter().copied().fold(f64::NEG_INFINITY, maximum),
                GlobalPoolKind::Average => elements.iter().sum::<f64>() / count_to_f64(plane),
            }
        })
        .collect();
    round_to_output(&pooled, output)
}
