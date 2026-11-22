//! Sliding-window geometry shared by convolution and pooling.

use super::TypeError;
use ptxc_core::{Dim, Rank, Shape};
use std::num::NonZeroU64;

/// The step between window positions along an axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Stride(NonZeroU64);

/// The spacing between window taps along an axis; 1 is a dense window.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Dilation(NonZeroU64);

/// Implicit elements added at one end of an axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Padding(u64);

/// Declares a constructor and accessor for a non-zero window newtype.
macro_rules! non_zero_extent {
    ($($name:ident),+) => {$(
        impl $name {
            /// The value, or `None` for zero, which ONNX does not allow.
            pub const fn new(value: u64) -> Option<Self> {
                match NonZeroU64::new(value) {
                    Some(value) => Some(Self(value)),
                    None => None,
                }
            }

            /// The value.
            pub const fn get(self) -> u64 {
                self.0.get()
            }
        }
    )+};
}

non_zero_extent!(Stride, Dilation);

impl Padding {
    /// Wraps a pad width.
    pub const fn new(width: u64) -> Self {
        Self(width)
    }

    /// The pad width.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// How a window moves along one spatial axis. Pads are explicit: the
/// importer resolves ONNX's `auto_pad`, which static shapes allow (D3).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WindowAxis {
    /// The step between positions.
    pub stride: Stride,
    /// The spacing between taps.
    pub dilation: Dilation,
    /// Padding before the first element.
    pub pad_begin: Padding,
    /// Padding after the last element.
    pub pad_end: Padding,
}

/// How a partial last window is treated, from ONNX's `ceil_mode`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Rounding {
    /// Only whole windows count.
    Floor,
    /// A partial last window counts, unless it would start in the end
    /// padding.
    Ceil,
}

/// The number of window positions along an axis, as ONNX's
/// `convPoolShapeInference` computes it.
pub(super) fn output_extent(
    input: Dim,
    kernel: Dim,
    axis: WindowAxis,
    rounding: Rounding,
) -> Result<Dim, TypeError> {
    let too_large = || TypeError::WindowTooLarge { input, kernel };
    let effective_kernel = kernel
        .get()
        .checked_sub(1)
        .ok_or(TypeError::EmptyKernel)?
        .checked_mul(axis.dilation.get())
        .and_then(|span| span.checked_add(1))
        .ok_or_else(too_large)?;
    let leading = input.get().checked_add(axis.pad_begin.get()).ok_or_else(too_large)?;
    let padded = leading.checked_add(axis.pad_end.get()).ok_or_else(too_large)?;
    let slack = padded.checked_sub(effective_kernel).ok_or_else(too_large)?;
    let stride = axis.stride.get();
    let positions = match rounding {
        Rounding::Floor => slack / stride + 1,
        Rounding::Ceil => {
            let positions = slack.div_ceil(stride) + 1;
            let last_start = (positions - 1).checked_mul(stride);
            let starts_in_end_padding = last_start.is_none_or(|start| start >= leading);
            if starts_in_end_padding { positions - 1 } else { positions }
        }
    };
    Ok(Dim::new(positions))
}

/// `[N, C, spatial...]` of a shape that must have `rank` (at least 2) dims.
pub(super) fn split_leading_two(
    shape: &Shape,
    rank: Rank,
) -> Result<(Dim, Dim, &[Dim]), TypeError> {
    match shape.dims() {
        [first, second, rest @ ..] if shape.rank() == rank => Ok((*first, *second, rest)),
        _ => Err(TypeError::Rank { expected: rank, actual: shape.rank() }),
    }
}
