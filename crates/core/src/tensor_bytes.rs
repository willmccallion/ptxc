//! Tensor elements as little-endian bytes, as device memory and program
//! files hold them: floats by their bits, booleans as one byte, 0 or 1.

use crate::dtype::DType;
use crate::float::{Bf16, F8E4M3Fn, F8E5M2, F16};
use crate::tensor::TensorData;
use thiserror::Error;

/// Bytes that are not a tensor of the expected type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum BytesError {
    /// The length is not a whole number of elements.
    #[error("{length} bytes are not a whole number of {dtype:?} elements")]
    Length {
        /// The element type.
        dtype: DType,
        /// The number of bytes.
        length: usize,
    },
    /// A boolean byte other than 0 or 1.
    #[error("boolean element {index} is the byte {byte}, not 0 or 1")]
    Bool {
        /// The element's position.
        index: usize,
        /// The byte.
        byte: u8,
    },
}

impl TensorData {
    /// The elements as little-endian bytes.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        match self {
            Self::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::F16(v) => v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
            Self::Bf16(v) => v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
            Self::F8E4M3Fn(v) => v.iter().map(|x| x.to_bits()).collect(),
            Self::F8E5M2(v) => v.iter().map(|x| x.to_bits()).collect(),
            Self::I8(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::U8(v) => v.clone(),
            Self::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Bool(v) => v.iter().map(|&x| u8::from(x)).collect(),
        }
    }

    /// Elements of `dtype` from their little-endian bytes.
    pub fn from_le_bytes(dtype: DType, bytes: &[u8]) -> Result<Self, BytesError> {
        Ok(match dtype {
            DType::F32 => decode(dtype, bytes, f32::from_le_bytes)?.into(),
            DType::F16 => decode(dtype, bytes, |b| F16::from_bits(u16::from_le_bytes(b)))?.into(),
            DType::Bf16 => decode(dtype, bytes, |b| Bf16::from_bits(u16::from_le_bytes(b)))?.into(),
            DType::F8E4M3Fn => decode(dtype, bytes, |[b]| F8E4M3Fn::from_bits(b))?.into(),
            DType::F8E5M2 => decode(dtype, bytes, |[b]| F8E5M2::from_bits(b))?.into(),
            DType::I8 => decode(dtype, bytes, i8::from_le_bytes)?.into(),
            DType::U8 => bytes.to_vec().into(),
            DType::I32 => decode(dtype, bytes, i32::from_le_bytes)?.into(),
            DType::I64 => decode(dtype, bytes, i64::from_le_bytes)?.into(),
            DType::Bool => decode_bools(bytes)?.into(),
        })
    }
}

/// Each `N`-byte element of `bytes`, decoded by `element`.
fn decode<const N: usize, T>(
    dtype: DType,
    bytes: &[u8],
    element: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>, BytesError> {
    let (elements, rest) = bytes.as_chunks::<N>();
    if !rest.is_empty() {
        return Err(BytesError::Length { dtype, length: bytes.len() });
    }
    Ok(elements.iter().map(|&chunk| element(chunk)).collect())
}

fn decode_bools(bytes: &[u8]) -> Result<Vec<bool>, BytesError> {
    bytes
        .iter()
        .enumerate()
        .map(|(index, &byte)| match byte {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(BytesError::Bool { index, byte }),
        })
        .collect()
}
