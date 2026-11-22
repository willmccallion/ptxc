//! Quantization parameters: `Scale`, `ZeroPoint` and per-tensor or per-axis
//! `QuantParams`, as ONNX's `QuantizeLinear` and `DequantizeLinear` take them.

use crate::dtype::DType;
use crate::shape::Axis;
use std::hash::{Hash, Hasher};
use std::ops::RangeInclusive;
use thiserror::Error;

/// Quantization parameters that ptxc does not accept.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum QuantError {
    /// A zero point outside the quantized type's range.
    #[error("zero point {} is out of range for {dtype:?}", .zero_point.get())]
    ZeroPointOutOfRange {
        /// The zero point.
        zero_point: ZeroPoint,
        /// The quantized type.
        dtype: QuantDType,
    },
}

/// The step between adjacent quantized values. Equality and hashing compare
/// bits, so a `Scale` can sit in an `Eq + Hash` op attribute.
#[derive(Clone, Copy, Debug)]
pub struct Scale(f32);

impl Scale {
    /// Wraps a scale.
    pub const fn new(scale: f32) -> Self {
        Self(scale)
    }

    /// The scale.
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl PartialEq for Scale {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for Scale {}

impl Hash for Scale {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// The quantized value that represents real zero.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ZeroPoint(i32);

impl ZeroPoint {
    /// Wraps a zero point.
    pub const fn new(zero_point: i32) -> Self {
        Self(zero_point)
    }

    /// The zero point.
    pub const fn get(self) -> i32 {
        self.0
    }
}

/// The element type of a quantized tensor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum QuantDType {
    /// Signed 8-bit integers.
    I8,
    /// Unsigned 8-bit integers.
    U8,
    /// Signed 32-bit integers, as QDQ models store matmul and conv biases.
    I32,
    /// FP8 E4M3FN.
    F8E4M3Fn,
    /// FP8 E5M2.
    F8E5M2,
}

impl QuantDType {
    /// The tensor element type.
    pub const fn dtype(self) -> DType {
        match self {
            Self::I8 => DType::I8,
            Self::U8 => DType::U8,
            Self::I32 => DType::I32,
            Self::F8E4M3Fn => DType::F8E4M3Fn,
            Self::F8E5M2 => DType::F8E5M2,
        }
    }

    /// The zero points ptxc accepts for this type.
    fn zero_point_range(self) -> RangeInclusive<i32> {
        match self {
            Self::I8 => i32::from(i8::MIN)..=i32::from(i8::MAX),
            Self::U8 => i32::from(u8::MIN)..=i32::from(u8::MAX),
            Self::I32 => i32::MIN..=i32::MAX,
            // NOTE: ONNX allows any FP8 value as an FP8 zero point. ptxc
            // accepts only 0, which is what ORT's quantizer and TensorRT
            // produce, so the zero point stays an integer for every type.
            Self::F8E4M3Fn | Self::F8E5M2 => 0..=0,
        }
    }
}

/// One affine map between real and quantized values:
/// `real = scale * (quantized - zero_point)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Affine {
    /// The step between adjacent quantized values.
    pub scale: Scale,
    /// The quantized value that represents real zero.
    pub zero_point: ZeroPoint,
}

/// Which elements share an affine map.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Granularity {
    /// One map for the whole tensor.
    PerTensor(Affine),
    /// One map per index along `axis`, in index order.
    PerAxis {
        /// The axis the maps vary along.
        axis: Axis,
        /// One map per index along `axis`.
        maps: Vec<Affine>,
    },
}

/// How a tensor is quantized: its quantized type and its affine maps.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct QuantParams {
    dtype: QuantDType,
    granularity: Granularity,
}

impl QuantParams {
    /// Parameters for `dtype`, checking every zero point against its range.
    pub fn new(dtype: QuantDType, granularity: Granularity) -> Result<Self, QuantError> {
        let maps = match &granularity {
            Granularity::PerTensor(map) => std::slice::from_ref(map),
            Granularity::PerAxis { maps, .. } => maps.as_slice(),
        };
        let range = dtype.zero_point_range();
        if let Some(map) = maps.iter().find(|map| !range.contains(&map.zero_point.0)) {
            return Err(QuantError::ZeroPointOutOfRange { zero_point: map.zero_point, dtype });
        }
        Ok(Self { dtype, granularity })
    }

    /// The quantized element type.
    pub const fn dtype(&self) -> QuantDType {
        self.dtype
    }

    /// Which elements share an affine map, and the maps.
    pub const fn granularity(&self) -> &Granularity {
        &self.granularity
    }
}
