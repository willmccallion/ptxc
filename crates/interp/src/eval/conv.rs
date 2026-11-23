//! Grouped N-d convolution, accumulating in f64; `QConv` shares the core.

use super::round_to_output;
use super::window::Windows;
use crate::error::{InterpError, Malformed};
use crate::extents::Extents;
use crate::numeric::wide_floats;
use ptxc_core::{HostTensor, TensorData};
use ptxc_graph::{ConvAttrs, TensorType, WindowAxis};

/// A grouped convolution of `x: [N, C, s…]` by `w: [M, C / groups, k…]`
/// into `[N, M, o…]`.
#[derive(Debug)]
pub(super) struct Convolution {
    input: Extents,
    weights: Extents,
    output: Extents,
    windows: Windows,
    taps: Vec<Vec<usize>>,
    groups: usize,
}

impl Convolution {
    pub(super) fn of(
        axes: &[WindowAxis],
        groups: u64,
        x: &HostTensor,
        w: &HostTensor,
        output: &TensorType,
    ) -> Result<Self, InterpError> {
        let (input, weights) = (Extents::of(x.shape())?, Extents::of(w.shape())?);
        let kernel = Extents::from_dims(weights.dims().get(2..).unwrap_or_default().to_vec());
        let taps = kernel.indices().collect();
        let windows = Windows::of(axes, &input)?;
        let groups =
            usize::try_from(groups).map_err(|_| InterpError::TooLarge { elements: groups })?;
        Ok(Self { output: Extents::of(output.shape())?, input, weights, windows, taps, groups })
    }

    pub(super) const fn output(&self) -> &Extents {
        &self.output
    }

    /// Each output element: the sum, over its group's input channels and
    /// every kernel tap inside the input, of `x * w`, folded with `mul_add`.
    pub(super) fn convolve<T: Copy>(
        &self,
        x: &[T],
        w: &[T],
        zero: T,
        mul_add: impl Fn(T, T, T) -> T,
    ) -> Vec<T> {
        self.output.indices().map(|out| self.element(x, w, &out, zero, &mul_add)).collect()
    }

    fn element<T: Copy>(
        &self,
        x: &[T],
        w: &[T],
        out: &[usize],
        zero: T,
        mul_add: &impl Fn(T, T, T) -> T,
    ) -> T {
        let (n, m, spatial) = (out[0], out[1], &out[2..]);
        let (xs, ws) = (self.input.strides(), self.weights.strides());
        let group_channels = self.weights.dims()[1];
        let group = m / (self.weights.dims()[0] / self.groups);
        let x_base = n * xs[0] + group * group_channels * xs[1];
        let mut acc = zero;
        for tap in &self.taps {
            let Some(source) = self.windows.source(spatial, tap, &xs[2..]) else { continue };
            let w_tap: usize = tap.iter().zip(&ws[2..]).map(|(t, s)| t * s).sum();
            for c in 0..group_channels {
                acc =
                    mul_add(x[x_base + c * xs[1] + source], w[m * ws[0] + c * ws[1] + w_tap], acc);
            }
        }
        acc
    }
}

/// `Conv(x, w, b?)` in f64, the bias added per output channel.
pub(super) fn conv(
    attrs: &ConvAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let (x, w, bias) = match operands {
        [x, w] => (x, w, None),
        [x, w, b] => (x, w, Some(wide_floats(b.data())?)),
        _ => return Err(Malformed::Operands.into()),
    };
    let convolution = Convolution::of(&attrs.axes, attrs.groups.get(), x, w, output)?;
    let sums =
        convolution.convolve(&wide_floats(x.data())?, &wide_floats(w.data())?, 0.0, |a, b, acc| {
            a.mul_add(b, acc)
        });
    let channels = convolution.output().strides()[1];
    let channel_count = convolution.output().dims()[1];
    let biased: Vec<f64> = sums
        .into_iter()
        .enumerate()
        .map(|(flat, sum)| sum + bias.as_ref().map_or(0.0, |b| b[flat / channels % channel_count]))
        .collect();
    round_to_output(&biased, output)
}
