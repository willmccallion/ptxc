//! Host values in each type a GEMM reads or writes, and back.

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
    use super::{HostError, data, values};
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
    fn integers_must_be_whole_and_in_range() {
        assert!(matches!(data(DType::I8, &[0.5]), Err(HostError::Integer { .. })));
        assert!(matches!(data(DType::I8, &[300.0]), Err(HostError::Integer { .. })));
        assert!(matches!(data(DType::Bool, &[1.0]), Err(HostError::Type(DType::Bool))));
    }
}
