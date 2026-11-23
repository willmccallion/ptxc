//! `TensorProto` to `HostTensor`, covering raw data, typed fields and external data.

use crate::error::{ImportError, ImportErrorKind, Unsupported};
use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16};
use ptxc_core::{DType, Dim, HostTensor, Shape, TensorData};
use ptxc_onnx_proto::TensorProto;
use ptxc_onnx_proto::tensor_proto::{DataLocation, DataType};
use std::path::{Component, Path};

/// The element type an ONNX `data_type` code names, if ptxc supports it.
pub(crate) fn dtype_from_onnx(code: i32) -> Result<DType, Unsupported> {
    let known = DataType::try_from(code).ok();
    match known {
        Some(DataType::Float) => Ok(DType::F32),
        Some(DataType::Float16) => Ok(DType::F16),
        Some(DataType::Bfloat16) => Ok(DType::Bf16),
        Some(DataType::Float8e4m3fn) => Ok(DType::F8E4M3Fn),
        Some(DataType::Float8e5m2) => Ok(DType::F8E5M2),
        Some(DataType::Int8) => Ok(DType::I8),
        Some(DataType::Uint8) => Ok(DType::U8),
        Some(DataType::Int32) => Ok(DType::I32),
        Some(DataType::Int64) => Ok(DType::I64),
        Some(DataType::Bool) => Ok(DType::Bool),
        Some(other) => Err(Unsupported::DType(other.as_str_name().to_owned())),
        None => Err(Unsupported::DType(format!("code {code}"))),
    }
}

/// Converts a tensor. External data is read relative to `model_dir`.
pub(crate) fn import_tensor(
    proto: &TensorProto,
    model_dir: Option<&Path>,
) -> Result<HostTensor, ImportError> {
    let name = proto.name.clone().unwrap_or_default();
    let dtype =
        dtype_from_onnx(proto.data_type.ok_or(ImportErrorKind::Missing("tensor data_type"))?)?;
    let shape = shape_from_dims(&proto.dims)?;
    let data = if proto.data_location == Some(DataLocation::External as i32) {
        decode_raw(dtype, &read_external(proto, model_dir)?)
    } else if let Some(raw) = &proto.raw_data {
        decode_raw(dtype, raw)
    } else {
        decode_typed(dtype, proto)?
    };
    let actual = with_len(&data);
    HostTensor::new(shape.clone(), data).map_err(|_| {
        ImportErrorKind::DataLength { name, expected: shape.element_count().get(), actual }.into()
    })
}

/// Imports a serialised `TensorProto`, such as an ONNX test case's
/// `input_0.pb`. Such files hold no external data.
pub fn tensor_from_bytes(bytes: &[u8]) -> Result<HostTensor, ImportError> {
    let proto = <TensorProto as prost::Message>::decode(bytes).map_err(ImportErrorKind::Decode)?;
    import_tensor(&proto, None)
}

/// A shape from ONNX dims, which must not be negative.
pub(crate) fn shape_from_dims(dims: &[i64]) -> Result<Shape, ImportError> {
    let dims = dims
        .iter()
        .map(|&dim| u64::try_from(dim).map(Dim::new))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid("tensor dims", format!("{dims:?} has a negative dimension")))?;
    Shape::new(dims).map_err(|error| invalid("tensor dims", error.to_string()))
}

fn invalid(field: &'static str, problem: String) -> ImportError {
    ImportErrorKind::Invalid { field, problem }.into()
}

fn with_len(data: &TensorData) -> u64 {
    let len = ptxc_core::with_tensor_data!(data, elements => elements.len());
    u64::try_from(len).unwrap_or(u64::MAX)
}

/// Little-endian bytes as elements; a trailing partial element is dropped,
/// which the length check then reports.
fn decode_raw(dtype: DType, bytes: &[u8]) -> TensorData {
    match dtype {
        DType::F32 => le::<4, _>(bytes, f32::from_le_bytes).into(),
        DType::F16 => le::<2, _>(bytes, |b| F16::from_bits(u16::from_le_bytes(b))).into(),
        DType::Bf16 => le::<2, _>(bytes, |b| Bf16::from_bits(u16::from_le_bytes(b))).into(),
        DType::F8E4M3Fn => bytes.iter().map(|&b| F8E4M3Fn::from_bits(b)).collect::<Vec<_>>().into(),
        DType::F8E5M2 => bytes.iter().map(|&b| F8E5M2::from_bits(b)).collect::<Vec<_>>().into(),
        DType::I8 => bytes.iter().map(|&b| i8::from_le_bytes([b])).collect::<Vec<_>>().into(),
        DType::U8 => bytes.to_vec().into(),
        DType::I32 => le::<4, _>(bytes, i32::from_le_bytes).into(),
        DType::I64 => le::<8, _>(bytes, i64::from_le_bytes).into(),
        DType::Bool => bytes.iter().map(|&b| b != 0).collect::<Vec<_>>().into(),
    }
}

fn le<const N: usize, T>(bytes: &[u8], from: impl Fn([u8; N]) -> T) -> Vec<T> {
    bytes.as_chunks::<N>().0.iter().map(|&chunk| from(chunk)).collect()
}

/// The typed fields: `float_data` for f32, `int64_data` for i64, and
/// `int32_data` for every narrower type, holding narrow floats as bits.
fn decode_typed(dtype: DType, proto: &TensorProto) -> Result<TensorData, ImportError> {
    let narrow = |what: &'static str| invalid("int32_data", format!("a value does not fit {what}"));
    Ok(match dtype {
        DType::F32 => proto.float_data.clone().into(),
        DType::I64 => proto.int64_data.clone().into(),
        DType::I32 => proto.int32_data.clone().into(),
        DType::I8 => {
            convert(&proto.int32_data, |v| i8::try_from(v).ok()).ok_or_else(|| narrow("i8"))?.into()
        }
        DType::U8 => {
            convert(&proto.int32_data, |v| u8::try_from(v).ok()).ok_or_else(|| narrow("u8"))?.into()
        }
        DType::Bool => convert(&proto.int32_data, |v| matches!(v, 0 | 1).then_some(v == 1))
            .ok_or_else(|| narrow("bool"))?
            .into(),
        DType::F16 => convert(&proto.int32_data, |v| u16::try_from(v).ok().map(F16::from_bits))
            .ok_or_else(|| narrow("f16 bits"))?
            .into(),
        DType::Bf16 => convert(&proto.int32_data, |v| u16::try_from(v).ok().map(Bf16::from_bits))
            .ok_or_else(|| narrow("bf16 bits"))?
            .into(),
        DType::F8E4M3Fn => {
            convert(&proto.int32_data, |v| u8::try_from(v).ok().map(F8E4M3Fn::from_bits))
                .ok_or_else(|| narrow("FP8 bits"))?
                .into()
        }
        DType::F8E5M2 => {
            convert(&proto.int32_data, |v| u8::try_from(v).ok().map(F8E5M2::from_bits))
                .ok_or_else(|| narrow("FP8 bits"))?
                .into()
        }
    })
}

fn convert<T>(values: &[i32], each: impl Fn(i32) -> Option<T>) -> Option<Vec<T>> {
    values.iter().map(|&value| each(value)).collect()
}

/// Reads the bytes an external-data tensor names: `location` relative to the
/// model's directory (and never outside it), then optional `offset` and
/// `length`.
fn read_external(proto: &TensorProto, model_dir: Option<&Path>) -> Result<Vec<u8>, ImportError> {
    let entry = |key: &str| {
        proto
            .external_data
            .iter()
            .find(|entry| entry.key.as_deref() == Some(key))
            .and_then(|entry| entry.value.clone())
    };
    let location = entry("location").ok_or(ImportErrorKind::Missing("external data location"))?;
    let relative = Path::new(&location);
    if !relative.components().all(|component| matches!(component, Component::Normal(_))) {
        return Err(invalid(
            "external data location",
            format!("{location:?} leaves the model's directory"),
        ));
    }
    let path = model_dir
        .ok_or(ImportErrorKind::Missing("the model's directory, for external data"))?
        .join(relative);
    let bytes = std::fs::read(&path)
        .map_err(|source| ImportErrorKind::ExternalData { path: path.clone(), source })?;
    let number = |key: &'static str| {
        entry(key)
            .map(|text| {
                text.parse::<usize>()
                    .map_err(|_| invalid("external data", format!("{key} {text:?} is not a size")))
            })
            .transpose()
    };
    let offset = number("offset")?.unwrap_or(0);
    let length = number("length")?.unwrap_or_else(|| bytes.len().saturating_sub(offset));
    let end = offset.checked_add(length).filter(|&end| end <= bytes.len());
    end.map(|end| bytes[offset..end].to_vec()).ok_or_else(|| {
        invalid(
            "external data",
            format!("{} has no bytes {offset}..{offset}+{length}", path.display()),
        )
    })
}
