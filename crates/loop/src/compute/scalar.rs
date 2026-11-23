//! `Scalar`: a typed constant in a tensor expression.

use ptxc_core::DType;
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
    /// A float constant of `dtype`, which must be a float type; the value
    /// is rounded to `dtype` where it is used.
    pub const fn float(dtype: DType, value: f64) -> Option<Self> {
        if is_float_storage(dtype) {
            Some(Self { dtype, value: ScalarValue::Float(value.to_bits()) })
        } else {
            None
        }
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

    /// A boolean constant.
    pub const fn bool(value: bool) -> Self {
        Self { dtype: DType::Bool, value: ScalarValue::Bool(value) }
    }

    /// The constant's element type.
    pub const fn dtype(self) -> DType {
        self.dtype
    }

    /// The value of a float constant.
    pub const fn as_f64(self) -> Option<f64> {
        match self.value {
            ScalarValue::Float(bits) => Some(f64::from_bits(bits)),
            _ => None,
        }
    }

    /// The value of an integer constant.
    pub const fn as_i64(self) -> Option<i64> {
        match self.value {
            ScalarValue::Int(value) => Some(value),
            _ => None,
        }
    }

    /// The value of a boolean constant.
    pub const fn as_bool(self) -> Option<bool> {
        match self.value {
            ScalarValue::Bool(value) => Some(value),
            _ => None,
        }
    }
}

/// Whether `dtype` stores floats, FP8 included.
const fn is_float_storage(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::Bf16 | DType::F8E4M3Fn | DType::F8E5M2)
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
