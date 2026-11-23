//! The `DType` enum: the element types ptxc supports (D5).

use crate::size::ByteSize;

/// A tensor element type.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum DType {
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary16.
    F16,
    /// bfloat16.
    Bf16,
    /// FP8 with 4 exponent and 3 mantissa bits, no infinities.
    F8E4M3Fn,
    /// FP8 with 5 exponent and 2 mantissa bits.
    F8E5M2,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Boolean, stored as one byte as in ONNX.
    Bool,
}

impl DType {
    /// Every element type, in declaration order.
    pub const ALL: [Self; 10] = [
        Self::F32,
        Self::F16,
        Self::Bf16,
        Self::F8E4M3Fn,
        Self::F8E5M2,
        Self::I8,
        Self::U8,
        Self::I32,
        Self::I64,
        Self::Bool,
    ];

    /// The type's name in ptxc's text formats, such as `f32` or `f8e4m3fn`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F8E4M3Fn => "f8e4m3fn",
            Self::F8E5M2 => "f8e5m2",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Bool => "bool",
        }
    }

    /// The type `name` names.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|dtype| dtype.name() == name)
    }

    /// The bytes one element occupies in device memory and in ONNX
    /// `raw_data`.
    pub const fn element_size(self) -> ByteSize {
        let bytes = match self {
            Self::F8E4M3Fn | Self::F8E5M2 | Self::I8 | Self::U8 | Self::Bool => 1,
            Self::F16 | Self::Bf16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
        };
        ByteSize::new(bytes)
    }
}
