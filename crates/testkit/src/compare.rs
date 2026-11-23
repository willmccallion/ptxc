//! Tolerance comparison per data type with a report of the worst mismatches.

use ptxc_core::{DType, ElementCount, HostTensor, Shape, TensorData};
use std::fmt;
use thiserror::Error;

/// How closely a result must match its reference.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tolerance {
    /// `|actual - reference| <= atol + rtol * |reference|` per element.
    Relative {
        /// The absolute allowance.
        atol: f64,
        /// The allowance relative to the reference.
        rtol: f64,
    },
    /// At most this many representable values apart: FP8 compared in its
    /// encoding, and quantized integers.
    Steps(u32),
    /// Equal.
    Exact,
}

impl Tolerance {
    /// The ONNX backend node tests' own tolerance.
    pub const ONNX_SUITE: Self = Self::Relative { atol: 1e-7, rtol: 1e-3 };

    /// Quantized `i8` and `u8` results, whose rounding ties can differ after
    /// rescaling.
    pub const QUANTIZED: Self = Self::Steps(1);

    /// The default for results of `dtype`, from TESTING.md's table.
    pub const fn for_dtype(dtype: DType) -> Self {
        match dtype {
            DType::F32 => Self::Relative { atol: 1e-5, rtol: 1e-4 },
            DType::F16 => Self::Relative { atol: 1e-3, rtol: 1e-2 },
            DType::Bf16 => Self::Relative { atol: 1e-2, rtol: 2e-2 },
            DType::F8E4M3Fn | DType::F8E5M2 => Self::Steps(1),
            DType::I8 | DType::U8 | DType::I32 | DType::I64 | DType::Bool => Self::Exact,
        }
    }

    /// The tolerance for a reduction of `length` terms: a relative tolerance
    /// grows by `sqrt(length)`, as rounding errors of independent terms do.
    #[must_use]
    pub fn for_reduction(self, length: ElementCount) -> Self {
        match self {
            Self::Relative { atol, rtol } => {
                let factor = wide(length.get()).sqrt();
                Self::Relative { atol: atol * factor, rtol: rtol * factor }
            }
            other => other,
        }
    }
}

#[expect(clippy::cast_precision_loss, reason = "a reduction length is far below 2^53")]
const fn wide(count: u64) -> f64 {
    count as f64
}

/// One element as the tolerances see it: its value, and its ordinal (the
/// integer itself, or a float's position in its encoding's order).
#[derive(Clone, Copy, Debug)]
struct Element {
    value: f64,
    ordinal: i64,
}

/// A float encoding's bits as a sign-magnitude ordinal: adjacent values are
/// one apart, and both zeros are 0.
fn ordinal(bits: u32, sign: u32) -> i64 {
    let magnitude = i64::from(bits & !sign);
    if bits & sign == 0 { magnitude } else { -magnitude }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the value is for error measures; exactness uses the ordinal"
)]
const fn approximate(value: i64) -> f64 {
    value as f64
}

fn elements(data: &TensorData) -> Vec<Element> {
    let float = |value: f64, bits: u32, sign: u32| Element { value, ordinal: ordinal(bits, sign) };
    let int = |value: i64| Element { value: approximate(value), ordinal: value };
    match data {
        TensorData::F32(v) => {
            v.iter().map(|&x| float(f64::from(x), x.to_bits(), 1 << 31)).collect()
        }
        TensorData::F16(v) => {
            v.iter().map(|x| float(x.to_f64(), u32::from(x.to_bits()), 1 << 15)).collect()
        }
        TensorData::Bf16(v) => {
            v.iter().map(|x| float(x.to_f64(), u32::from(x.to_bits()), 1 << 15)).collect()
        }
        TensorData::F8E4M3Fn(v) => {
            v.iter().map(|x| float(x.to_f64(), u32::from(x.to_bits()), 1 << 7)).collect()
        }
        TensorData::F8E5M2(v) => {
            v.iter().map(|x| float(x.to_f64(), u32::from(x.to_bits()), 1 << 7)).collect()
        }
        TensorData::I8(v) => v.iter().map(|&x| int(i64::from(x))).collect(),
        TensorData::U8(v) => v.iter().map(|&x| int(i64::from(x))).collect(),
        TensorData::I32(v) => v.iter().map(|&x| int(i64::from(x))).collect(),
        TensorData::I64(v) => v.iter().map(|&x| int(x)).collect(),
        TensorData::Bool(v) => v.iter().map(|&x| int(i64::from(x))).collect(),
    }
}

impl Tolerance {
    /// Whether `actual` is close enough to `expected`. Two NaNs match, and an
    /// infinity matches only itself.
    fn accepts(self, actual: Element, expected: Element) -> bool {
        if actual.value.is_nan() || expected.value.is_nan() {
            return actual.value.is_nan() && expected.value.is_nan();
        }
        match self {
            Self::Relative { .. } if actual.value.is_infinite() || expected.value.is_infinite() => {
                actual.ordinal == expected.ordinal
            }
            Self::Relative { atol, rtol } => {
                (actual.value - expected.value).abs() <= rtol.mul_add(expected.value.abs(), atol)
            }
            Self::Steps(steps) => actual.ordinal.abs_diff(expected.ordinal) <= u64::from(steps),
            Self::Exact => actual.ordinal == expected.ordinal,
        }
    }
}

/// An element's position in its tensor, one index per axis.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ElementPosition(pub Vec<u64>);

impl ElementPosition {
    fn of(flat: usize, shape: &Shape) -> Self {
        let mut remaining = count(flat);
        let mut index: Vec<u64> = shape
            .dims()
            .iter()
            .rev()
            .map(|dim| {
                let extent = dim.get();
                let i = remaining % extent;
                remaining /= extent;
                i
            })
            .collect();
        index.reverse();
        Self(index)
    }
}

impl fmt::Display for ElementPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.0.iter().map(u64::to_string).collect();
        write!(f, "[{}]", parts.join(", "))
    }
}

/// The element with the largest error by one measure.
#[derive(Clone, PartialEq, Debug)]
pub struct Worst {
    /// Where it is.
    pub position: ElementPosition,
    /// The result's value.
    pub actual: f64,
    /// The reference's value.
    pub expected: f64,
    /// Its error by the measure: infinite where exactly one is NaN.
    pub error: f64,
}

impl fmt::Display for Worst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at {}: {} vs expected {}",
            self.error, self.position, self.actual, self.expected
        )
    }
}

/// How a result's values differ from the reference's.
#[derive(Clone, PartialEq, Debug)]
pub struct ValueReport {
    /// Elements outside the tolerance.
    pub mismatched: ElementCount,
    /// All elements.
    pub total: ElementCount,
    /// The largest `|actual - expected|`.
    pub worst_absolute: Worst,
    /// The largest `|actual - expected| / |expected|`.
    pub worst_relative: Worst,
}

impl fmt::Display for ValueReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} of {} elements differ; worst absolute error {}; worst relative error {}",
            self.mismatched.get(),
            self.total.get(),
            self.worst_absolute,
            self.worst_relative
        )
    }
}

/// Why a result does not match its reference.
#[derive(Clone, PartialEq, Debug, Error)]
pub enum Mismatch {
    /// The element types differ.
    #[error("element type {actual:?}, expected {expected:?}")]
    DType {
        /// The result's.
        actual: DType,
        /// The reference's.
        expected: DType,
    },
    /// The shapes differ.
    #[error("shape {:?}, expected {:?}", .actual.dims(), .expected.dims())]
    Shape {
        /// The result's.
        actual: Shape,
        /// The reference's.
        expected: Shape,
    },
    /// Some elements are outside the tolerance.
    #[error("{0}")]
    Values(ValueReport),
}

/// Compares a result with its reference under `tolerance`.
pub fn compare(
    actual: &HostTensor,
    expected: &HostTensor,
    tolerance: Tolerance,
) -> Result<(), Mismatch> {
    if actual.dtype() != expected.dtype() {
        return Err(Mismatch::DType { actual: actual.dtype(), expected: expected.dtype() });
    }
    if actual.shape() != expected.shape() {
        return Err(Mismatch::Shape {
            actual: actual.shape().clone(),
            expected: expected.shape().clone(),
        });
    }
    let pairs: Vec<(Element, Element)> =
        elements(actual.data()).into_iter().zip(elements(expected.data())).collect();
    let mismatched = pairs.iter().filter(|(a, e)| !tolerance.accepts(*a, *e)).count();
    if mismatched == 0 {
        return Ok(());
    }
    Err(Mismatch::Values(report(&pairs, mismatched, expected.shape())))
}

/// The report of a comparison in which some of `pairs`, so at least one,
/// mismatched.
fn report(pairs: &[(Element, Element)], mismatched: usize, shape: &Shape) -> ValueReport {
    let worst = |measure: fn(f64, f64) -> f64| {
        let error = |flat: usize| measure(pairs[flat].0.value, pairs[flat].1.value);
        let flat = (1..pairs.len())
            .fold(0, |best, flat| if error(flat) > error(best) { flat } else { best });
        let (actual, expected) = pairs[flat];
        Worst {
            position: ElementPosition::of(flat, shape),
            actual: actual.value,
            expected: expected.value,
            error: error(flat),
        }
    };
    ValueReport {
        mismatched: ElementCount::new(count(mismatched)),
        total: ElementCount::new(count(pairs.len())),
        worst_absolute: worst(absolute_error),
        worst_relative: worst(relative_error),
    }
}

/// A count of elements in memory as `u64`, which holds any `usize`.
const fn count(elements: usize) -> u64 {
    elements as u64
}

/// `|actual - expected|`; infinite where exactly one is NaN, 0 where both are.
fn absolute_error(actual: f64, expected: f64) -> f64 {
    match (actual.is_nan(), expected.is_nan()) {
        (true, true) => 0.0,
        (true, false) | (false, true) => f64::INFINITY,
        (false, false) if actual.total_cmp(&expected).is_eq() => 0.0,
        (false, false) => (actual - expected).abs(),
    }
}

/// `|actual - expected| / |expected|`; infinite for any error from zero.
fn relative_error(actual: f64, expected: f64) -> f64 {
    let absolute = absolute_error(actual, expected);
    if absolute == 0.0 { 0.0 } else { absolute / expected.abs() }
}
