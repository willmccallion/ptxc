//! Building `ModelProto`s by hand.

use prost::Message;
use ptxc_graph::Graph;
use ptxc_graph::text::print;
use ptxc_onnx::{ImportError, ImportOptions, import_model};
use ptxc_onnx_proto::attribute_proto::AttributeType;
use ptxc_onnx_proto::tensor_proto::DataType;
use ptxc_onnx_proto::tensor_shape_proto::{Dimension, dimension};
use ptxc_onnx_proto::type_proto::{Tensor, Value};
use ptxc_onnx_proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto,
};

pub(crate) const FLOAT: DataType = DataType::Float;
pub(crate) const INT64: DataType = DataType::Int64;

/// A declared dimension: fixed, or symbolic by name.
pub(crate) enum D {
    N(i64),
    S(&'static str),
}

pub(crate) fn value_info(name: &str, dtype: DataType, dims: &[D]) -> ValueInfoProto {
    let dim = dims
        .iter()
        .map(|dim| Dimension {
            value: Some(match dim {
                D::N(value) => dimension::Value::DimValue(*value),
                D::S(name) => dimension::Value::DimParam((*name).to_owned()),
            }),
            ..Dimension::default()
        })
        .collect();
    let tensor = Tensor { elem_type: Some(dtype as i32), shape: Some(TensorShapeProto { dim }) };
    ValueInfoProto {
        name: Some(name.to_owned()),
        r#type: Some(TypeProto { value: Some(Value::TensorType(tensor)), ..TypeProto::default() }),
        ..ValueInfoProto::default()
    }
}

pub(crate) fn fixed(name: &str, dtype: DataType, dims: &[i64]) -> ValueInfoProto {
    value_info(name, dtype, &dims.iter().map(|&d| D::N(d)).collect::<Vec<_>>())
}

pub(crate) fn node(
    op: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Vec<AttributeProto>,
) -> NodeProto {
    NodeProto {
        op_type: Some(op.to_owned()),
        input: inputs.iter().map(|&name| name.to_owned()).collect(),
        output: outputs.iter().map(|&name| name.to_owned()).collect(),
        attribute: attrs,
        ..NodeProto::default()
    }
}

fn attr(name: &str, kind: AttributeType) -> AttributeProto {
    AttributeProto {
        name: Some(name.to_owned()),
        r#type: Some(kind as i32),
        ..AttributeProto::default()
    }
}

pub(crate) fn int(name: &str, value: i64) -> AttributeProto {
    AttributeProto { i: Some(value), ..attr(name, AttributeType::Int) }
}

pub(crate) fn ints(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto { ints: values.to_vec(), ..attr(name, AttributeType::Ints) }
}

pub(crate) fn float(name: &str, value: f32) -> AttributeProto {
    AttributeProto { f: Some(value), ..attr(name, AttributeType::Float) }
}

pub(crate) fn string(name: &str, value: &str) -> AttributeProto {
    AttributeProto { s: Some(value.as_bytes().to_vec()), ..attr(name, AttributeType::String) }
}

pub(crate) fn floats(name: &str, values: &[f32]) -> AttributeProto {
    AttributeProto { floats: values.to_vec(), ..attr(name, AttributeType::Floats) }
}

pub(crate) fn tensor_attr(name: &str, tensor: TensorProto) -> AttributeProto {
    AttributeProto { t: Some(tensor), ..attr(name, AttributeType::Tensor) }
}

pub(crate) fn i64_tensor(name: &str, dims: &[i64], values: &[i64]) -> TensorProto {
    TensorProto {
        name: Some(name.to_owned()),
        data_type: Some(INT64 as i32),
        dims: dims.to_vec(),
        int64_data: values.to_vec(),
        ..TensorProto::default()
    }
}

pub(crate) fn f32_tensor(name: &str, dims: &[i64], values: &[f32]) -> TensorProto {
    TensorProto {
        name: Some(name.to_owned()),
        data_type: Some(FLOAT as i32),
        dims: dims.to_vec(),
        float_data: values.to_vec(),
        ..TensorProto::default()
    }
}

/// A model at `opset` with these nodes, inputs, initializers and outputs.
pub(crate) fn model_at(
    opset: i64,
    nodes: Vec<NodeProto>,
    inputs: Vec<ValueInfoProto>,
    initializers: Vec<TensorProto>,
    outputs: Vec<ValueInfoProto>,
) -> ModelProto {
    let graph = GraphProto {
        node: nodes,
        input: inputs,
        initializer: initializers,
        output: outputs,
        ..GraphProto::default()
    };
    ModelProto {
        ir_version: Some(9),
        opset_import: vec![OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(opset),
        }],
        graph: Some(graph),
        ..ModelProto::default()
    }
}

pub(crate) fn model(
    nodes: Vec<NodeProto>,
    inputs: Vec<ValueInfoProto>,
    initializers: Vec<TensorProto>,
    outputs: Vec<ValueInfoProto>,
) -> ModelProto {
    model_at(21, nodes, inputs, initializers, outputs)
}

pub(crate) fn import(model: &ModelProto) -> Result<Graph, ImportError> {
    import_with(model, &ImportOptions::default())
}

pub(crate) fn import_with(
    model: &ModelProto,
    options: &ImportOptions,
) -> Result<Graph, ImportError> {
    import_model(&model.encode_to_vec(), options)
}

/// The imported graph's text.
pub(crate) fn text(model: &ModelProto) -> String {
    let graph = import(model).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(graph.verify(), Ok(()));
    print(&graph)
}

/// The text of the single op line in the imported graph's text, without
/// its result names.
pub(crate) fn op_line(model: &ModelProto) -> String {
    let text = text(model);
    let line = text
        .lines()
        .find(|line| line.contains(" = ") && !line.contains(" = const "))
        .unwrap_or_else(|| panic!("no op in\n{text}"));
    line.split_once(" = ").unwrap().1.to_owned()
}

pub(crate) const BOOL: DataType = DataType::Bool;
pub(crate) const INT32: DataType = DataType::Int32;

/// The op line of a one-node model at opset 21.
pub(crate) fn line(
    op: NodeProto,
    inputs: Vec<ValueInfoProto>,
    constants: Vec<TensorProto>,
) -> String {
    line_at(21, op, inputs, constants)
}

pub(crate) fn line_at(
    opset: i64,
    op: NodeProto,
    inputs: Vec<ValueInfoProto>,
    constants: Vec<TensorProto>,
) -> String {
    op_line(&model_at(opset, vec![op], inputs, constants, vec![]))
}

/// Compares each `(actual, expected)` pair, reporting every mismatch at once.
pub(crate) fn assert_lines(cases: &[(String, &str)]) {
    let wrong: Vec<String> = cases
        .iter()
        .filter(|(actual, expected)| actual != expected)
        .map(|(actual, expected)| format!("expected {expected}\n     got {actual}"))
        .collect();
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

/// The printed definition of the graph output `name`: a model whose only
/// output is `output`, imported, and the text line that defines it.
pub(crate) fn output_line(
    nodes: Vec<NodeProto>,
    inputs: Vec<ValueInfoProto>,
    output: ValueInfoProto,
) -> String {
    let name = output.name.clone().unwrap();
    let text = text(&model(nodes, inputs, vec![], vec![output]));
    let prefix = format!("%{name} = ");
    let line = text.lines().map(str::trim).find(|line| line.starts_with(&prefix));
    line.unwrap_or_else(|| panic!("no definition of {name} in\n{text}"))[prefix.len()..].to_owned()
}

/// A node's input names: `data`, then each constant's name.
pub(crate) fn input_names<'a>(data: &'a str, constants: &'a [TensorProto]) -> Vec<&'a str> {
    std::iter::once(data)
        .chain(constants.iter().map(|constant| constant.name.as_deref().unwrap()))
        .collect()
}
