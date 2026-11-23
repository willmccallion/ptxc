//! Sliding windows over spatial axes, shared by convolution and pooling.

use crate::error::InterpError;
use crate::extents::Extents;
use ptxc_graph::WindowAxis;

/// One spatial axis of a sliding window, in `usize`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Window {
    stride: usize,
    dilation: usize,
    pad_begin: usize,
    pad_end: usize,
    input: usize,
}

impl Window {
    fn of(axis: &WindowAxis, input: usize) -> Result<Self, InterpError> {
        let size = |value: u64| {
            usize::try_from(value).map_err(|_| InterpError::TooLarge { elements: value })
        };
        Ok(Self {
            stride: size(axis.stride.get())?,
            dilation: size(axis.dilation.get())?,
            pad_begin: size(axis.pad_begin.get())?,
            pad_end: size(axis.pad_end.get())?,
            input,
        })
    }

    /// The position in the padded input that tap `tap` of output `out` reads.
    const fn padded_position(&self, out: usize, tap: usize) -> usize {
        out * self.stride + tap * self.dilation
    }

    /// The input element that tap reads, `None` where it reads padding.
    fn source(&self, out: usize, tap: usize) -> Option<usize> {
        self.padded_position(out, tap).checked_sub(self.pad_begin).filter(|&i| i < self.input)
    }

    /// Whether the tap lies within the input or its explicit padding; a
    /// `ceil_mode` window can reach past both.
    const fn within_padding(&self, out: usize, tap: usize) -> bool {
        self.padded_position(out, tap) < self.pad_begin + self.input + self.pad_end
    }
}

/// The windows of every spatial axis of an `[N, C, spatial…]` input.
#[derive(Debug)]
pub(super) struct Windows {
    axes: Vec<Window>,
}

impl Windows {
    pub(super) fn of(axes: &[WindowAxis], input: &Extents) -> Result<Self, InterpError> {
        let spatial = input.dims().get(2..).unwrap_or_default();
        let axes = axes
            .iter()
            .zip(spatial)
            .map(|(axis, &extent)| Window::of(axis, extent))
            .collect::<Result<_, _>>()?;
        Ok(Self { axes })
    }

    /// The offset, within one `[n, c]` plane of the input, that a tap of an
    /// output position reads; `None` where any axis reads padding.
    pub(super) fn source(&self, out: &[usize], tap: &[usize], strides: &[usize]) -> Option<usize> {
        let mut offset = 0;
        for (((window, &o), &t), &stride) in self.axes.iter().zip(out).zip(tap).zip(strides) {
            offset += window.source(o, t)? * stride;
        }
        Some(offset)
    }

    /// Whether every axis of the tap lies within the input or its padding.
    pub(super) fn within_padding(&self, out: &[usize], tap: &[usize]) -> bool {
        self.axes.iter().zip(out).zip(tap).all(|((window, &o), &t)| window.within_padding(o, t))
    }
}
