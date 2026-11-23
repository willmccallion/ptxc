//! `Value`: one element as the interpreter holds it, always exactly
//! representable in its element type.

use crate::compute::Scalar;
use ptxc_core::DType;
use ptxc_core::convert::{f64_to_f32, int_to_f64};
use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16, Overflow};

/// An element: floats as the `f64` of their exact value, integers wrapped
/// to their width, booleans as themselves.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Value {
    Float(f64),
    Int(i64),
    Bool(bool),
}

impl Value {
    /// A constant, rounded to its type.
    pub(crate) fn of_scalar(scalar: Scalar) -> Self {
        let dtype = scalar.dtype();
        match (scalar.as_f64(), scalar.as_i64(), scalar.as_bool()) {
            (Some(float), _, _) => round(float, dtype, Overflow::Saturate),
            (_, Some(int), _) => Self::Int(int),
            (_, _, Some(flag)) => Self::Bool(flag),
            (None, None, None) => Self::Bool(false),
        }
    }

    /// The value as a float: integers converted, booleans 0 or 1.
    pub(crate) const fn float(self) -> f64 {
        match self {
            Self::Float(value) => value,
            Self::Int(value) => int_to_f64(value),
            Self::Bool(value) => {
                if value {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// The value as an integer: floats truncated, booleans 0 or 1.
    pub(crate) const fn int(self) -> i64 {
        match self {
            Self::Float(value) => ptxc_core::convert::truncate(value),
            Self::Int(value) => value,
            Self::Bool(value) => {
                if value {
                    1
                } else {
                    0
                }
            }
        }
    }

    /// The value as a boolean: whether it is non-zero.
    pub(crate) fn bool(self) -> bool {
        match self {
            Self::Float(value) => value != 0.0,
            Self::Int(value) => value != 0,
            Self::Bool(value) => value,
        }
    }
}

/// `value` rounded to the float type `dtype`, nearest with ties to even;
/// FP8 overflow follows `overflow`.
pub(crate) fn round(value: f64, dtype: DType, overflow: Overflow) -> Value {
    let rounded = match dtype {
        DType::F32 => f64::from(f64_to_f32(value)),
        DType::F16 => F16::from_f64(value).to_f64(),
        DType::Bf16 => Bf16::from_f64(value).to_f64(),
        DType::F8E4M3Fn => F8E4M3Fn::from_f64(value, overflow).to_f64(),
        DType::F8E5M2 => F8E5M2::from_f64(value, overflow).to_f64(),
        DType::I8 | DType::U8 | DType::I32 | DType::I64 => {
            return wrap(i128::from(ptxc_core::convert::truncate(value)), dtype);
        }
        DType::Bool => return Value::Bool(value != 0.0),
    };
    Value::Float(rounded)
}

/// `value` wrapped to the integer type `dtype`, as two's complement does.
pub(crate) fn wrap(value: i128, dtype: DType) -> Value {
    let bits = match dtype {
        DType::I8 | DType::U8 => 8,
        DType::I32 => 32,
        _ => 64,
    };
    let modulus = 1_i128 << bits;
    let unsigned = value.rem_euclid(modulus);
    let wrapped =
        if dtype != DType::U8 && unsigned >= modulus / 2 { unsigned - modulus } else { unsigned };
    Value::Int(i64::try_from(wrapped).unwrap_or(i64::MIN))
}

/// `value` clamped to the integer type `dtype`'s range.
pub(crate) fn saturate(value: i64, dtype: DType) -> Value {
    let (low, high) = match dtype {
        DType::I8 => (-128, 127),
        DType::U8 => (0, 255),
        DType::I32 => (i64::from(i32::MIN), i64::from(i32::MAX)),
        _ => (i64::MIN, i64::MAX),
    };
    Value::Int(value.clamp(low, high))
}
