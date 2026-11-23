//! Numeric semantics (D19): elements widen to f32, i64 or bool for each op;
//! matmul, convolution and reductions accumulate floats in f64; results
//! round to their storage type, floats to nearest-even and integers by
//! wrapping, as NumPy narrows.

use crate::error::{InterpError, Malformed};
use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16, Overflow};
use ptxc_core::{DType, TensorData};

/// A tensor's elements in the domain an op computes in.
#[derive(Clone, Debug)]
pub(crate) enum Elements {
    /// Every float type, widened exactly to f32.
    Float(Vec<f32>),
    /// Every integer type, widened exactly to i64.
    Int(Vec<i64>),
    /// Bools.
    Bool(Vec<bool>),
}

/// The elements of `data` in their computing domain; widening is exact.
pub(crate) fn widen(data: &TensorData) -> Elements {
    match data {
        TensorData::F32(values) => Elements::Float(values.clone()),
        TensorData::F16(values) => Elements::Float(values.iter().map(|v| v.to_f32()).collect()),
        TensorData::Bf16(values) => Elements::Float(values.iter().map(|v| v.to_f32()).collect()),
        TensorData::F8E4M3Fn(values) => {
            Elements::Float(values.iter().map(|v| v.to_f32()).collect())
        }
        TensorData::F8E5M2(values) => Elements::Float(values.iter().map(|v| v.to_f32()).collect()),
        TensorData::I8(values) => Elements::Int(values.iter().map(|&v| i64::from(v)).collect()),
        TensorData::U8(values) => Elements::Int(values.iter().map(|&v| i64::from(v)).collect()),
        TensorData::I32(values) => Elements::Int(values.iter().map(|&v| i64::from(v)).collect()),
        TensorData::I64(values) => Elements::Int(values.clone()),
        TensorData::Bool(values) => Elements::Bool(values.clone()),
    }
}

/// A float tensor's elements as f32; other types are not floats.
pub(crate) fn floats(data: &TensorData) -> Result<Vec<f32>, InterpError> {
    match widen(data) {
        Elements::Float(values) => Ok(values),
        Elements::Int(_) | Elements::Bool(_) => Err(Malformed::DType(data.dtype()).into()),
    }
}

/// An integer tensor's elements as i64.
pub(crate) fn ints(data: &TensorData) -> Result<Vec<i64>, InterpError> {
    match widen(data) {
        Elements::Int(values) => Ok(values),
        Elements::Float(_) | Elements::Bool(_) => Err(Malformed::DType(data.dtype()).into()),
    }
}

/// A float tensor's elements as f64, for accumulation.
pub(crate) fn wide_floats(data: &TensorData) -> Result<Vec<f64>, InterpError> {
    Ok(floats(data)?.into_iter().map(f64::from).collect())
}

/// Elements stored as `dtype`, which must be in their domain.
pub(crate) fn store(elements: Elements, dtype: DType) -> Result<TensorData, InterpError> {
    match elements {
        Elements::Float(values) => store_floats(&values, dtype, Overflow::NonFinite),
        Elements::Int(values) => store_ints(&values, dtype),
        Elements::Bool(values) => Ok(values.into()),
    }
}

/// Floats rounded to nearest-even in `dtype`; `overflow` decides what an
/// out-of-range value becomes in an FP8 type.
pub(crate) fn store_floats(
    values: &[f32],
    dtype: DType,
    overflow: Overflow,
) -> Result<TensorData, InterpError> {
    Ok(match dtype {
        DType::F32 => values.to_vec().into(),
        DType::F16 => values.iter().map(|&v| F16::from_f32(v)).collect::<Vec<_>>().into(),
        DType::Bf16 => values.iter().map(|&v| Bf16::from_f32(v)).collect::<Vec<_>>().into(),
        DType::F8E4M3Fn => {
            values.iter().map(|&v| F8E4M3Fn::from_f32(v, overflow)).collect::<Vec<_>>().into()
        }
        DType::F8E5M2 => {
            values.iter().map(|&v| F8E5M2::from_f32(v, overflow)).collect::<Vec<_>>().into()
        }
        _ => return Err(Malformed::DType(dtype).into()),
    })
}

/// Accumulated or exact values rounded once, to nearest-even, into `dtype`;
/// `overflow` decides what an out-of-range value becomes in an FP8 type.
pub(crate) fn store_wide(
    values: &[f64],
    dtype: DType,
    overflow: Overflow,
) -> Result<TensorData, InterpError> {
    Ok(match dtype {
        DType::F32 => values.iter().map(|&v| f64_to_f32(v)).collect::<Vec<_>>().into(),
        DType::F16 => values.iter().map(|&v| F16::from_f64(v)).collect::<Vec<_>>().into(),
        DType::Bf16 => values.iter().map(|&v| Bf16::from_f64(v)).collect::<Vec<_>>().into(),
        DType::F8E4M3Fn => {
            values.iter().map(|&v| F8E4M3Fn::from_f64(v, overflow)).collect::<Vec<_>>().into()
        }
        DType::F8E5M2 => {
            values.iter().map(|&v| F8E5M2::from_f64(v, overflow)).collect::<Vec<_>>().into()
        }
        _ => return Err(Malformed::DType(dtype).into()),
    })
}

/// Integers narrowed to `dtype` by wrapping, as NumPy narrows.
pub(crate) fn store_ints(values: &[i64], dtype: DType) -> Result<TensorData, InterpError> {
    Ok(match dtype {
        DType::I64 => values.to_vec().into(),
        DType::I32 => values.iter().map(|&v| wrap_i32(v)).collect::<Vec<_>>().into(),
        DType::I8 => values.iter().map(|&v| wrap_i8(v)).collect::<Vec<_>>().into(),
        DType::U8 => values.iter().map(|&v| wrap_u8(v)).collect::<Vec<_>>().into(),
        _ => return Err(Malformed::DType(dtype).into()),
    })
}

/// Integers saturated to `dtype`'s range.
pub(crate) fn saturate_ints(values: &[i64], dtype: DType) -> Result<TensorData, InterpError> {
    let (low, high) = int_range(dtype)?;
    store_ints(&values.iter().map(|&v| v.clamp(low, high)).collect::<Vec<_>>(), dtype)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "f64 to f32 rounds to nearest-even, as D19 asks"
)]
pub(crate) const fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

/// `value` truncated toward zero; out of range saturates and NaN gives 0.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Rust's float-to-int cast truncates and saturates"
)]
pub(crate) const fn truncate(value: f64) -> i64 {
    value as i64
}

/// An integer as the nearest f64; exact below 2^53.
#[expect(clippy::cast_precision_loss, reason = "rounds to nearest-even, as ONNX Cast does")]
pub(crate) const fn int_to_f64(value: i64) -> f64 {
    value as f64
}

/// An integer as the nearest f32, rounded once.
#[expect(clippy::cast_precision_loss, reason = "rounds to nearest-even, as ONNX Cast does")]
pub(crate) const fn int_to_f32(value: i64) -> f32 {
    value as f32
}

#[expect(clippy::cast_possible_truncation, reason = "wrapping is how NumPy narrows integers")]
const fn wrap_i32(value: i64) -> i32 {
    value as i32
}

#[expect(clippy::cast_possible_truncation, reason = "wrapping is how NumPy narrows integers")]
const fn wrap_i8(value: i64) -> i8 {
    value as i8
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "wrapping is how NumPy narrows integers"
)]
const fn wrap_u8(value: i64) -> u8 {
    value as u8
}

/// An element count as the nearest f32, for averages.
#[expect(
    clippy::cast_precision_loss,
    reason = "counts are far below 2^24 in practice; rounds to nearest"
)]
pub(crate) const fn count_to_f32(count: usize) -> f32 {
    count as f32
}

/// An element count as the nearest f64, for averages.
#[expect(clippy::cast_precision_loss, reason = "exact below 2^53")]
pub(crate) const fn count_to_f64(count: usize) -> f64 {
    count as f64
}

/// The lowest and highest values of an integer type, widened.
pub(crate) fn int_range(dtype: DType) -> Result<(i64, i64), InterpError> {
    Ok(match dtype {
        DType::I64 => (i64::MIN, i64::MAX),
        DType::I32 => (i32::MIN.into(), i32::MAX.into()),
        DType::I8 => (i8::MIN.into(), i8::MAX.into()),
        DType::U8 => (u8::MIN.into(), u8::MAX.into()),
        _ => return Err(Malformed::DType(dtype).into()),
    })
}

/// A position within a tensor as i64, as ONNX's index outputs hold.
#[expect(clippy::cast_possible_wrap, reason = "positions within a tensor are below 2^63")]
pub(crate) const fn position_to_i64(position: usize) -> i64 {
    position as i64
}

/// A position as i128, wide enough for signed arithmetic on any position.
pub(crate) const fn signed(position: usize) -> i128 {
    position as i128
}

/// The larger value, NaN if either is NaN, as NumPy's `maximum`.
pub(crate) const fn maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
}

/// The smaller value, NaN if either is NaN, as NumPy's `minimum`.
pub(crate) const fn minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) }
}
