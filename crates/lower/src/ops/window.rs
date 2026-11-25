//! Sliding windows over spatial axes, shared by convolution and pooling: where
//! each kernel tap reads, the input padded where it reads outside, and how
//! many taps of a window are counted.

use super::NodeContext;
use super::shape::shifted;
use crate::builder::{Operand, StageBuilder};
use crate::error::LowerError;
use crate::stage::Binding;
use ptxc_core::Dim;
use ptxc_graph::{ValueId, WindowAxis};
use ptxc_loop::{ExprArena, ExprId, Scalar};

/// One spatial axis of a window: the input extent, the window's output
/// extent and kernel extent, and how it moves.
#[derive(Clone, Copy)]
pub(super) struct Span {
    pub(super) input: Dim,
    pub(super) output: Dim,
    pub(super) kernel: Dim,
    pub(super) axis: WindowAxis,
}

/// Where tap `tap` of output position `out` reads along one axis: the
/// input position, negative or past the end in padding, and whether any tap
/// of any window reads padding.
pub(super) struct Read {
    pub(super) index: ExprId,
    pub(super) reads_padding: bool,
}

/// The integer constants of `span`, for index arithmetic.
struct Constants {
    input: i64,
    output: i64,
    kernel: i64,
    stride: i64,
    dilation: i64,
    pad_begin: i64,
    pad_end: i64,
}

impl Constants {
    fn of(cx: &NodeContext<'_>, span: Span) -> Result<Self, LowerError> {
        let signed = |value: u64| cx.signed(Dim::new(value));
        Ok(Self {
            input: cx.signed(span.input)?,
            output: cx.signed(span.output)?,
            kernel: cx.signed(span.kernel)?,
            stride: signed(span.axis.stride.get())?,
            dilation: signed(span.axis.dilation.get())?,
            pad_begin: signed(span.axis.pad_begin.get())?,
            pad_end: signed(span.axis.pad_end.get())?,
        })
    }

    /// Whether every tap of every window lies inside the input.
    const fn always_inside(&self) -> bool {
        let last_start = (self.output - 1).saturating_mul(self.stride);
        let last_tap = (self.kernel - 1).saturating_mul(self.dilation);
        self.pad_begin == 0 && last_start.saturating_add(last_tap) < self.input
    }
}

/// `index * factor`, leaving out a factor of 1.
fn scaled(exprs: &mut ExprArena, index: ExprId, factor: i64) -> ExprId {
    if factor == 1 {
        return index;
    }
    let factor = exprs.constant(factor);
    exprs.mul(index, factor)
}

/// `index / divisor`, floored, leaving out a divisor of 1.
fn divided(exprs: &mut ExprArena, index: ExprId, divisor: i64) -> ExprId {
    if divisor == 1 {
        return index;
    }
    let divisor = exprs.constant(divisor);
    exprs.floor_div(index, divisor)
}

/// `out * stride + tap * dilation - pad_begin`: the input position a tap
/// reads, negative or past the end in padding.
fn input_position(
    exprs: &mut ExprArena,
    constants: &Constants,
    (out, tap): (ExprId, ExprId),
) -> ExprId {
    let start = scaled(exprs, out, constants.stride);
    let offset = scaled(exprs, tap, constants.dilation);
    let padded = exprs.add(start, offset);
    shifted(exprs, padded, -constants.pad_begin)
}

/// Where tap `tap` of output `out` reads along `span`.
pub(super) fn read(
    cx: &NodeContext<'_>,
    exprs: &mut ExprArena,
    span: Span,
    (out, tap): (ExprId, ExprId),
) -> Result<Read, LowerError> {
    let constants = Constants::of(cx, span)?;
    let index = input_position(exprs, &constants, (out, tap));
    Ok(Read { index, reads_padding: !constants.always_inside() })
}

/// `input` read at `index`, a window's taps, or `padding` where a tap reads
/// padding, if any of `reads` does.
pub(super) fn window_load(
    stage: &mut StageBuilder<'_>,
    (input, index): (ValueId, Vec<ExprId>),
    reads: &[Read],
    padding: Scalar,
) -> Operand {
    if reads.iter().any(|read| read.reads_padding) {
        stage.padded_load(Binding::Value(input), index, padding)
    } else {
        stage.load(Binding::Value(input), index)
    }
}

/// Whether a window's taps count padding when averaging.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Counted {
    /// Taps inside the input only.
    Input,
    /// Taps inside the input or its explicit padding.
    Padded,
}

/// How many taps of output position `out`'s window along `span` are
/// counted: those `t` in `[0, kernel)` whose position lies in the counted
/// range, in closed form so no second reduction is needed.
pub(super) fn tap_count(
    cx: &NodeContext<'_>,
    exprs: &mut ExprArena,
    span: Span,
    out: ExprId,
    counted: Counted,
) -> Result<ExprId, LowerError> {
    let constants = Constants::of(cx, span)?;
    if constants.always_inside() {
        return Ok(exprs.constant(constants.kernel));
    }
    let (first, last) = match counted {
        Counted::Input => (constants.pad_begin, constants.pad_begin + constants.input - 1),
        Counted::Padded => (0, constants.pad_begin + constants.input + constants.pad_end - 1),
    };
    // The counted taps t satisfy first <= start + t * dilation <= last, so
    // ceil((first - start) / dilation) <= t <= floor((last - start) / dilation),
    // where ceil(a / d) is -floor(-a / d).
    let start = scaled(exprs, out, constants.stride);
    let past_first = shifted(exprs, start, -first);
    let floored = divided(exprs, past_first, constants.dilation);
    let first_tap = scaled(exprs, floored, -1);
    let zero = exprs.constant(0);
    let first_tap = exprs.max(first_tap, zero);
    let negated_start = scaled(exprs, out, -constants.stride);
    let before_last = shifted(exprs, negated_start, last);
    let last_tap = divided(exprs, before_last, constants.dilation);
    let kernel_last = exprs.constant(constants.kernel - 1);
    let last_tap = exprs.min(last_tap, kernel_last);
    let negated_first = scaled(exprs, first_tap, -1);
    let difference = exprs.add(last_tap, negated_first);
    let count = shifted(exprs, difference, 1);
    Ok(exprs.max(count, zero))
}
