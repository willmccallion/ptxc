//! A graph as an ONNX model: its inputs, its nodes in order and its
//! outputs, each value typed.

use super::nodes::node;
use super::{ExportError, dtype_to_onnx, onnx_dims};
use prost::Message;
use ptxc_core::Idx;
use ptxc_graph::{Graph, ValueDef, ValueId};
use ptxc_onnx_proto::tensor_shape_proto::{Dimension, dimension};
use ptxc_onnx_proto::type_proto::{Tensor, Value};
use ptxc_onnx_proto::{
    GraphProto, ModelProto, OperatorSetIdProto, TensorShapeProto, TypeProto, ValueInfoProto,
};

/// The default-domain opset models are written at.
const OPSET: i64 = 21;

/// The ONNX IR version that opset 21 belongs to.
const IR_VERSION: i64 = 10;

/// `graph` as the bytes of an ONNX model, if every op in it has an export.
pub fn model_to_bytes(graph: &Graph) -> Result<Vec<u8>, ExportError> {
    let mut proto = GraphProto { name: Some("ptxc".to_owned()), ..GraphProto::default() };
    for &input in graph.inputs() {
        proto.input.push(value_info(graph, input)?);
    }
    for (_, found) in graph.nodes() {
        let exported = node(
            found.op(),
            names(graph, found.operands().iter())?,
            names(graph, found.outputs().iter())?,
        )?;
        proto.node.push(exported.node);
        proto.initializer.extend(exported.initializers);
    }
    for &output in graph.outputs() {
        proto.output.push(value_info(graph, output)?);
    }
    let model = ModelProto {
        ir_version: Some(IR_VERSION),
        opset_import: vec![OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(OPSET),
        }],
        producer_name: Some("ptxc".to_owned()),
        graph: Some(proto),
        ..ModelProto::default()
    };
    Ok(model.encode_to_vec())
}

/// Each of `values`' names.
fn names<'g>(
    graph: &Graph,
    values: impl Iterator<Item = &'g ValueId>,
) -> Result<Vec<String>, ExportError> {
    values.map(|&value| value_name(graph, value)).collect()
}

/// `value`'s name, or one made from its id if it has none.
fn value_name(graph: &Graph, value: ValueId) -> Result<String, ExportError> {
    let found = graph.value(value);
    if let ValueDef::Constant(_) = found.def() {
        return Err(ExportError::Constant);
    }
    Ok(found.name().map_or_else(|| format!("_{}", value.index()), |name| name.as_str().to_owned()))
}

/// `value`'s name and tensor type.
fn value_info(graph: &Graph, value: ValueId) -> Result<ValueInfoProto, ExportError> {
    let ty = graph.value(value).ty();
    let dim = onnx_dims(ty.shape())?
        .into_iter()
        .map(|extent| Dimension {
            value: Some(dimension::Value::DimValue(extent)),
            ..Dimension::default()
        })
        .collect();
    let tensor = Tensor {
        elem_type: Some(dtype_to_onnx(ty.dtype()) as i32),
        shape: Some(TensorShapeProto { dim }),
    };
    Ok(ValueInfoProto {
        name: Some(value_name(graph, value)?),
        r#type: Some(TypeProto { value: Some(Value::TensorType(tensor)), ..TypeProto::default() }),
        ..ValueInfoProto::default()
    })
}
