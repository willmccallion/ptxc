//! Export to ONNX: a `HostTensor` as a serialised `TensorProto`, the
//! inverse of `tensor_from_bytes`, and a graph of the ops `nodes` maps as a
//! model.

mod graph;
mod nodes;

pub use graph::model_to_bytes;

use ptxc_core::{DType, HostTensor, Shape};
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

/// A tensor or graph ptxc cannot write as ONNX.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ExportError {
    /// A dimension past ONNX's `int64` dims.
    #[error("dimension {0} is too large for ONNX")]
    Dim(u64),
    /// An op the exporter has no ONNX node for.
    #[error("{0} has no ONNX export")]
    Op(String),
    /// An axis or count past ONNX's `int64` attributes.
    #[error("{0} is too large for ONNX")]
    Index(usize),
    /// A graph holding constants, which the exporter does not write.
    #[error("graphs with constants have no ONNX export")]
    Constant,
}

/// `tensor`, named `name`, as the bytes of a `TensorProto`, such as an ONNX
/// test case's `output_0.pb`.
pub fn tensor_to_bytes(tensor: &HostTensor, name: &str) -> Result<Vec<u8>, ExportError> {
    let proto = TensorProto {
        dims: onnx_dims(tensor.shape())?,
        data_type: Some(dtype_to_onnx(tensor.dtype()) as i32),
        raw_data: Some(tensor.data().to_le_bytes()),
        name: Some(name.to_owned()),
        ..TensorProto::default()
    };
    Ok(prost::Message::encode_to_vec(&proto))
}

/// `shape`'s extents as ONNX's `int64` dims.
fn onnx_dims(shape: &Shape) -> Result<Vec<i64>, ExportError> {
    shape
        .dims()
        .iter()
        .map(|dim| i64::try_from(dim.get()).map_err(|_| ExportError::Dim(dim.get())))
        .collect()
}
