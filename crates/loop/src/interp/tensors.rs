//! Host tensors in and out of interpreter memory.

use super::value::Value;
use ptxc_core::convert::f64_to_f32;
use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16, Overflow};
use ptxc_core::{DType, TensorData};

/// Each element of `data` as a value.
pub(super) fn values(data: &TensorData) -> Vec<Value> {
    match data {
        TensorData::F32(v) => v.iter().map(|&x| Value::Float(f64::from(x))).collect(),
        TensorData::F16(v) => v.iter().map(|x| Value::Float(x.to_f64())).collect(),
        TensorData::Bf16(v) => v.iter().map(|x| Value::Float(x.to_f64())).collect(),
        TensorData::F8E4M3Fn(v) => v.iter().map(|x| Value::Float(x.to_f64())).collect(),
        TensorData::F8E5M2(v) => v.iter().map(|x| Value::Float(x.to_f64())).collect(),
        TensorData::I8(v) => v.iter().map(|&x| Value::Int(i64::from(x))).collect(),
        TensorData::U8(v) => v.iter().map(|&x| Value::Int(i64::from(x))).collect(),
        TensorData::I32(v) => v.iter().map(|&x| Value::Int(i64::from(x))).collect(),
        TensorData::I64(v) => v.iter().map(|&x| Value::Int(x)).collect(),
        TensorData::Bool(v) => v.iter().map(|&x| Value::Bool(x)).collect(),
    }
}

/// `values`, each exactly representable in `dtype`, as tensor data. The
/// interpreter only stores values rounded or wrapped to their buffer's
/// type, so every conversion is exact.
pub(super) fn data(values: &[Value], dtype: DType) -> TensorData {
    let floats = || values.iter().map(|value| value.float());
    let ints = || values.iter().map(|value| value.int());
    match dtype {
        DType::F32 => floats().map(f64_to_f32).collect::<Vec<_>>().into(),
        DType::F16 => floats().map(F16::from_f64).collect::<Vec<_>>().into(),
        DType::Bf16 => floats().map(Bf16::from_f64).collect::<Vec<_>>().into(),
        DType::F8E4M3Fn => {
            floats().map(|x| F8E4M3Fn::from_f64(x, Overflow::NonFinite)).collect::<Vec<_>>().into()
        }
        DType::F8E5M2 => {
            floats().map(|x| F8E5M2::from_f64(x, Overflow::NonFinite)).collect::<Vec<_>>().into()
        }
        DType::I8 => ints().map(|x| i8::try_from(x).unwrap_or_default()).collect::<Vec<_>>().into(),
        DType::U8 => ints().map(|x| u8::try_from(x).unwrap_or_default()).collect::<Vec<_>>().into(),
        DType::I32 => {
            ints().map(|x| i32::try_from(x).unwrap_or_default()).collect::<Vec<_>>().into()
        }
        DType::I64 => ints().collect::<Vec<_>>().into(),
        DType::Bool => values.iter().map(|value| value.bool()).collect::<Vec<_>>().into(),
    }
}
