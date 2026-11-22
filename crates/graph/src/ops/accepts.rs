//! The element-type classes op type rules accept.

use super::TypeError;
use ptxc_core::DType;

/// A class of element types. FP8 is storage only: it reaches arithmetic
/// through `Cast` or dequantization, as ONNX's arithmetic ops require.
#[derive(Clone, Copy, Debug)]
pub(super) enum Accepts {
    /// Floats and integers.
    Numeric,
    /// Floats and signed integers.
    SignedNumeric,
    /// f32, f16 and bf16.
    Float,
    /// Floats, i8 and u8: what ONNX `MaxPool` takes.
    FloatOrInt8,
    /// Signed and unsigned integers.
    Integer,
    /// Floats, i32 and i64: what ONNX `Pow` bases and `PRelu` take.
    FloatOrWideInt,
    /// Bool only.
    Bool,
    /// Numeric types and bool.
    NumericOrBool,
}

impl Accepts {
    pub(super) const fn admits(self, dtype: DType) -> bool {
        let float = matches!(dtype, DType::F32 | DType::F16 | DType::Bf16);
        let signed = matches!(dtype, DType::I8 | DType::I32 | DType::I64);
        let unsigned = matches!(dtype, DType::U8);
        let boolean = matches!(dtype, DType::Bool);
        match self {
            Self::Numeric => float || signed || unsigned,
            Self::SignedNumeric => float || signed,
            Self::Float => float,
            Self::FloatOrInt8 => float || matches!(dtype, DType::I8 | DType::U8),
            Self::Integer => signed || unsigned,
            Self::FloatOrWideInt => float || matches!(dtype, DType::I32 | DType::I64),
            Self::Bool => boolean,
            Self::NumericOrBool => float || signed || unsigned || boolean,
        }
    }

    pub(super) const fn check(self, dtype: DType) -> Result<(), TypeError> {
        if self.admits(dtype) { Ok(()) } else { Err(TypeError::UnsupportedDType(dtype)) }
    }
}
