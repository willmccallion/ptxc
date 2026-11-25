//! Host values in each type a GEMM or convolution reads or writes, and
//! back, and the values a problem's inputs hold.

use ptxc_core::convert::{f64_to_f32, truncate};
use ptxc_core::float::{Bf16, F8E4M3Fn, F16, Overflow};
use ptxc_core::{DType, TensorData};
use thiserror::Error;

/// A value a GEMM's types cannot hold exactly as given.
#[derive(Clone, Copy, Debug, PartialEq, Error)]
pub(crate) enum HostError {
    #[error("{0:?} is not a type GEMMs read or write")]
    Type(DType),
    #[error("{value} is not an integer in {dtype:?}'s range")]
    Integer { value: f64, dtype: DType },
}

/// `values` as `dtype`'s elements, rounded to nearest; E4M3 saturates.
pub(crate) fn data(dtype: DType, values: &[f64]) -> Result<TensorData, HostError> {
    let each = values.iter().copied();
    Ok(match dtype {
        DType::F32 => each.map(f64_to_f32).collect::<Vec<_>>().into(),
        DType::F16 => each.map(F16::from_f64).collect::<Vec<_>>().into(),
        DType::Bf16 => each.map(Bf16::from_f64).collect::<Vec<_>>().into(),
        DType::F8E4M3Fn => {
            each.map(|v| F8E4M3Fn::from_f64(v, Overflow::Saturate)).collect::<Vec<_>>().into()
        }
        DType::I8 => each.map(|v| integer::<i8>(v, dtype)).collect::<Result<Vec<_>, _>>()?.into(),
        DType::I32 => each.map(|v| integer::<i32>(v, dtype)).collect::<Result<Vec<_>, _>>()?.into(),
        other => return Err(HostError::Type(other)),
    })
}

/// `data`'s elements as f64s, which hold every GEMM type exactly.
pub(crate) fn values(data: &TensorData) -> Result<Vec<f64>, HostError> {
    Ok(match data {
        TensorData::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        TensorData::F16(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::Bf16(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::F8E4M3Fn(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::I8(v) => v.iter().map(|&x| f64::from(x)).collect(),
        TensorData::I32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        other => return Err(HostError::Type(other.dtype())),
    })
}

/// FNV-1a's 64-bit offset basis and prime, which hash a problem's id.
const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// The salt of the problem named `id`: its hash, so no two problems'
/// inputs share their values.
pub(crate) fn salt_of(id: &str) -> u64 {
    id.bytes().fold(FNV_OFFSET, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME))
}

/// The value at row-major position `index` of the input salted `salt`: a
/// whole number in [-8, 8], spread by a hash of both.
pub(crate) fn whole_value(index: u64, salt: u64) -> f64 {
    let mixed = index.wrapping_add(salt.wrapping_mul(0x5851_F42D_4C95_7F2D));
    // The top five bits, below 32, so the low byte holds them exactly.
    let bucket = (mixed.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 59).to_le_bytes()[0];
    f64::from(bucket % 17) - 8.0
}

/// `value` as the integer type `T`, if it is one of its integers.
fn integer<T: TryFrom<i64>>(value: f64, dtype: DType) -> Result<T, HostError> {
    let fail = || HostError::Integer { value, dtype };
    if value.fract() != 0.0 || !(-9.0e15..=9.0e15).contains(&value) {
        return Err(fail());
    }
    T::try_from(truncate(value)).map_err(|_| fail())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{HostError, data, salt_of, values};
    use ptxc_core::DType;

    #[test]
    fn every_gemm_type_round_trips_its_values() {
        let numbers = [-2.0, -0.125, 0.0, 0.875, 3.0];
        for dtype in [DType::F32, DType::F16, DType::Bf16, DType::F8E4M3Fn] {
            assert_eq!(values(&data(dtype, &numbers).unwrap()).unwrap(), numbers, "{dtype:?}");
        }
        let integers = [-8.0, 0.0, 7.0];
        for dtype in [DType::I8, DType::I32] {
            assert_eq!(values(&data(dtype, &integers).unwrap()).unwrap(), integers, "{dtype:?}");
        }
    }

    #[test]
    fn a_problems_salt_is_the_fnv_1a_hash_of_its_name() {
        // FNV-1a's 64-bit test vectors: the empty string and "a".
        assert_eq!(salt_of(""), 0xCBF2_9CE4_8422_2325);
        assert_eq!(salt_of("a"), 0xAF63_DC4C_8601_EC8C);
    }

    #[test]
    fn integers_must_be_whole_and_in_range() {
        assert!(matches!(data(DType::I8, &[0.5]), Err(HostError::Integer { .. })));
        assert!(matches!(data(DType::I8, &[300.0]), Err(HostError::Integer { .. })));
        assert!(matches!(data(DType::Bool, &[1.0]), Err(HostError::Type(DType::Bool))));
    }
}
