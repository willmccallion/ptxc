//! Comparison with the error bounds of f32 sums, for results whose terms
//! may cancel.

use super::{Element, ElementPosition, Mismatch, count, elements};
use crate::families::values::floats;
use ptxc_core::{ElementCount, HostTensor};
use std::fmt;

/// Where the exact values of f32 sums lie, before their last rounding into a
/// result's type: within `γₙ Σ|xᵢ|` of `estimate`.
///
/// `n` counts the f32 roundings on any term's way into its sum, and
/// `γₙ = nu / (1 - nu)` for f32's unit roundoff `u` (Higham, *Accuracy and
/// Stability of Numerical Algorithms*, §3.1 and §4.2). The bound holds in
/// any order of summation, and however far the terms cancel, where no
/// tolerance relative to the result can.
#[derive(Clone, Copy, Debug)]
pub struct SumBound<'a> {
    /// A value near each exact sum, in any float type.
    pub estimate: &'a HostTensor,
    /// The magnitudes of each sum's terms, summed, `Σ|xᵢ|`; its sign is
    /// ignored.
    pub magnitude: &'a HostTensor,
    /// The f32 roundings on any term's way into its sum.
    pub roundings: u32,
}

impl SumBound<'_> {
    /// `γₙ` for `n` f32 roundings.
    fn gamma(self) -> f64 {
        let nu = f64::from(self.roundings) * f64::from(f32::EPSILON) / 2.0;
        nu / (1.0 - nu)
    }

    /// Each sum's interval, as its low and its high ends: only the estimate
    /// itself where it is NaN or infinite.
    fn ends(self) -> (Vec<f64>, Vec<f64>) {
        let gamma = self.gamma();
        let magnitudes = elements(self.magnitude.data());
        elements(self.estimate.data())
            .iter()
            .zip(magnitudes)
            .map(|(estimate, magnitude)| {
                if !estimate.value.is_finite() {
                    return (estimate.value, estimate.value);
                }
                let error = gamma * magnitude.value.abs();
                (estimate.value - error, estimate.value + error)
            })
            .unzip()
    }
}

/// The elements outside the intervals their sums' error bounds give.
#[derive(Clone, PartialEq, Debug)]
pub struct BoundsReport {
    /// Elements outside their intervals.
    pub outside: ElementCount,
    /// All elements.
    pub total: ElementCount,
    /// Where the first one outside is.
    pub position: ElementPosition,
    /// Its value.
    pub actual: f64,
    /// The low end of its interval, rounded into the result's type.
    pub low: f64,
    /// The high end, rounded likewise.
    pub high: f64,
}

impl fmt::Display for BoundsReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} of {} elements lie outside their sums' error bounds; the first, at {}, is {}, \
             not in [{}, {}]",
            self.outside.get(),
            self.total.get(),
            self.position,
            self.actual,
            self.low,
            self.high
        )
    }
}

/// Compares a float result with the sums `bound` describes.
///
/// Each element must lie between its interval's ends rounded to nearest
/// into the result's type (saturating, for FP8), as rounding to nearest
/// keeps every value of the interval between them. Where the interval is
/// NaN, so must the element be.
///
/// # Panics
///
/// If `actual` is not of a float type.
pub fn compare_bounded(actual: &HostTensor, bound: SumBound<'_>) -> Result<(), Mismatch> {
    for wide in [bound.estimate, bound.magnitude] {
        if wide.shape() != actual.shape() {
            return Err(Mismatch::Shape {
                actual: actual.shape().clone(),
                expected: wide.shape().clone(),
            });
        }
    }
    let rounded = |ends: &[f64]| elements(&floats(actual.dtype(), ends));
    let (low, high) = bound.ends();
    let (low, high) = (rounded(&low), rounded(&high));
    let values = elements(actual.data());
    let outside: Vec<usize> =
        (0..values.len()).filter(|&flat| !within(values[flat], low[flat], high[flat])).collect();
    let Some(&first) = outside.first() else { return Ok(()) };
    Err(Mismatch::Bounds(BoundsReport {
        outside: ElementCount::new(count(outside.len())),
        total: ElementCount::new(count(values.len())),
        position: ElementPosition::of(first, actual.shape()),
        actual: values[first].value,
        low: low[first].value,
        high: high[first].value,
    }))
}

/// Whether `value` lies between `low` and `high`, or is NaN where they are.
fn within(value: Element, low: Element, high: Element) -> bool {
    if low.value.is_nan() {
        return value.value.is_nan();
    }
    low.value <= value.value && value.value <= high.value
}
