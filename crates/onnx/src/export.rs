//! `HostTensor` to a serialised `TensorProto`, the inverse of
//! `tensor_from_bytes`: dims, the element type and raw little-endian data.

use ptxc_core::{DType, HostTensor};
use ptxc_onnx_proto::TensorProto;
use ptxc_onnx_proto::tensor_proto::DataType;
use thiserror::Error;

/// ONNX's code for `dtype`.
const fn dtype_to_onnx(dtype: DType) -> DataType {
    match dtype {
        DType::F32 => DataType::Float,
        DType::F16 => DataType::Float16,
        DType::Bf16 => DataType::Bfloat16,
        DType::F8E4M3Fn => DataType::Float8e4m3fn,
        DType::F8E5M2 => DataType::Float8e5m2,
        DType::I8 => DataType::Int8,
        DType::U8 => DataType::Uint8,
        DType::I32 => DataType::Int32,
        DType::I64 => DataType::Int64,
        DType::Bool => DataType::Bool,
    }
}

/// A tensor ONNX cannot hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum ExportError {
    /// A dimension past ONNX's `int64` dims.
    #[error("dimension {0} is too large for ONNX")]
    Dim(u64),
}

/// `tensor`, named `name`, as the bytes of a `TensorProto`, such as an ONNX
/// test case's `output_0.pb`.
pub fn tensor_to_bytes(tensor: &HostTensor, name: &str) -> Result<Vec<u8>, ExportError> {
    let dims = tensor
        .shape()
        .dims()
        .iter()
        .map(|dim| i64::try_from(dim.get()).map_err(|_| ExportError::Dim(dim.get())))
        .collect::<Result<_, _>>()?;
    let proto = TensorProto {
        dims,
        data_type: Some(dtype_to_onnx(tensor.dtype()) as i32),
        raw_data: Some(tensor.data().to_le_bytes()),
        name: Some(name.to_owned()),
        ..TensorProto::default()
    };
    Ok(prost::Message::encode_to_vec(&proto))
}
