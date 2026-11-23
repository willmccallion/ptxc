//! `Scalar`: a typed constant in a tensor expression.

use ptxc_core::DType;
use ptxc_core::convert::f64_to_f32;
use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16, Overflow};
use std::fmt;

/// A constant of one element type. Floats are kept as `f64` bits, so
/// constants compare and hash exactly, NaN included.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Scalar {
    dtype: DType,
    value: ScalarValue,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum ScalarValue {
    Float(u64),
    Int(i64),
    Bool(bool),
}

impl Scalar {
    /// A float constant of `dtype`, which must be a float type, rounded to
    /// it: nearest with ties to even, an FP8 value beyond the largest
    /// finite one saturating.
    pub fn float(dtype: DType, value: f64) -> Option<Self> {
        let rounded = round_to(dtype, value)?;
        Some(Self { dtype, value: ScalarValue::Float(rounded.to_bits()) })
    }

    /// An integer constant of `dtype`, which must be an integer type that
    /// holds `value`.
    pub fn int(dtype: DType, value: i64) -> Option<Self> {
        let fits = match dtype {
            DType::I8 => i8::try_from(value).is_ok(),
            DType::U8 => u8::try_from(value).is_ok(),
            DType::I32 => i32::try_from(value).is_ok(),
            DType::I64 => true,
            _ => false,
        };
        fits.then_some(Self { dtype, value: ScalarValue::Int(value) })
    }

    /// An `i64` constant.
    pub const fn i64(value: i64) -> Self {
        Self { dtype: DType::I64, value: ScalarValue::Int(value) }
    }

    /// An `f32` constant.
    pub fn f32(value: f32) -> Self {
        Self { dtype: DType::F32, value: ScalarValue::Float(f64::from(value).to_bits()) }
    }

    /// A boolean constant.
    pub const fn bool(value: bool) -> Self {
        Self { dtype: DType::Bool, value: ScalarValue::Bool(value) }
    }

    /// Zero of `dtype` (`false` for booleans).
    pub const fn zero(dtype: DType) -> Self {
        Self::extreme(dtype, 0.0, 0, false)
    }

    /// One of `dtype` (`true` for booleans).
    pub const fn one(dtype: DType) -> Self {
        Self::extreme(dtype, 1.0, 1, true)
    }

    /// The lowest value of `dtype`: negative infinity where the format has
    /// one, its most negative finite value where it does not (E4M3FN).
    pub const fn lowest(dtype: DType) -> Self {
        match dtype {
            DType::F8E4M3Fn => Self::extreme(dtype, -448.0, 0, false),
            DType::I8 => Self::extreme(dtype, 0.0, i8::MIN as i64, false),
            DType::U8 => Self::extreme(dtype, 0.0, 0, false),
            DType::I32 => Self::extreme(dtype, 0.0, i32::MIN as i64, false),
            _ => Self::extreme(dtype, f64::NEG_INFINITY, i64::MIN, false),
        }
    }

    /// The highest value of `dtype`: positive infinity where the format has
    /// one, its largest finite value where it does not (E4M3FN).
    pub const fn highest(dtype: DType) -> Self {
        match dtype {
            DType::F8E4M3Fn => Self::extreme(dtype, 448.0, 0, true),
            DType::I8 => Self::extreme(dtype, 0.0, i8::MAX as i64, true),
            DType::U8 => Self::extreme(dtype, 0.0, u8::MAX as i64, true),
            DType::I32 => Self::extreme(dtype, 0.0, i32::MAX as i64, true),
            _ => Self::extreme(dtype, f64::INFINITY, i64::MAX, true),
        }
    }

    /// `float` for float types, `int` for integers and `bool` for booleans;
    /// each caller passes values every type of its class holds.
    const fn extreme(dtype: DType, float: f64, int: i64, bool: bool) -> Self {
        let value = match dtype {
            DType::F32 | DType::F16 | DType::Bf16 | DType::F8E4M3Fn | DType::F8E5M2 => {
                ScalarValue::Float(float.to_bits())
            }
            DType::I8 | DType::U8 | DType::I32 | DType::I64 => ScalarValue::Int(int),
            DType::Bool => ScalarValue::Bool(bool),
        };
        Self { dtype, value }
    }

    /// The constant's element type.
    pub const fn dtype(self) -> DType {
        self.dtype
    }

    /// The value, whichever kind it is.
    pub const fn constant(self) -> Constant {
        match self.value {
            ScalarValue::Float(bits) => Constant::Float(f64::from_bits(bits)),
            ScalarValue::Int(value) => Constant::Int(value),
            ScalarValue::Bool(value) => Constant::Bool(value),
        }
    }
}

/// A constant's value: a float or an integer its type holds exactly, or a
/// boolean.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Constant {
    /// A float.
    Float(f64),
    /// An integer.
    Int(i64),
    /// A boolean.
    Bool(bool),
}

/// `value` rounded to the float type `dtype`, or `None` if `dtype` is not
/// a float type.
fn round_to(dtype: DType, value: f64) -> Option<f64> {
    Some(match dtype {
        DType::F32 => f64::from(f64_to_f32(value)),
        DType::F16 => F16::from_f64(value).to_f64(),
        DType::Bf16 => Bf16::from_f64(value).to_f64(),
        DType::F8E4M3Fn => F8E4M3Fn::from_f64(value, Overflow::Saturate).to_f64(),
        DType::F8E5M2 => F8E5M2::from_f64(value, Overflow::Saturate).to_f64(),
        DType::I8 | DType::U8 | DType::I32 | DType::I64 | DType::Bool => return None,
    })
}

impl fmt::Display for Scalar {
    /// `1.5:f32`, `-3:i64` or `true`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value {
            ScalarValue::Float(bits) => {
                write!(f, "{:?}:{}", f64::from_bits(bits), self.dtype.name())
            }
            ScalarValue::Int(value) => write!(f, "{value}:{}", self.dtype.name()),
            ScalarValue::Bool(value) => write!(f, "{value}"),
        }
    }
}
