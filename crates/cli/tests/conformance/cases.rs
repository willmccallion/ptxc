//! Node test cases written to disk: one per outcome.

use prost::Message;
use ptxc_onnx_proto::attribute_proto::AttributeType;
use ptxc_onnx_proto::tensor_proto::DataType;
use ptxc_onnx_proto::tensor_shape_proto::{Dimension, dimension};
use ptxc_onnx_proto::type_proto::{Sequence, Tensor, Value};
use ptxc_onnx_proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto,
};
use std::path::Path;

fn tensor_type(dtype: DataType, dims: &[i64]) -> TypeProto {
    let dim = dims
        .iter()
        .map(|&d| Dimension { value: Some(dimension::Value::DimValue(d)), ..Dimension::default() })
        .collect();
    let tensor = Tensor { elem_type: Some(dtype as i32), shape: Some(TensorShapeProto { dim }) };
    TypeProto { value: Some(Value::TensorType(tensor)), ..TypeProto::default() }
}

fn info(name: &str, ty: TypeProto) -> ValueInfoProto {
    ValueInfoProto { name: Some(name.to_owned()), r#type: Some(ty), ..ValueInfoProto::default() }
}

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        op_type: Some(op.to_owned()),
        input: inputs.iter().map(|&i| i.to_owned()).collect(),
        output: outputs.iter().map(|&o| o.to_owned()).collect(),
        ..NodeProto::default()
    }
}

fn model(
    nodes: Vec<NodeProto>,
    inputs: Vec<ValueInfoProto>,
    outputs: Vec<ValueInfoProto>,
) -> ModelProto {
    let graph = GraphProto { node: nodes, input: inputs, output: outputs, ..GraphProto::default() };
    ModelProto {
        ir_version: Some(9),
        opset_import: vec![OperatorSetIdProto { domain: Some(String::new()), version: Some(21) }],
        graph: Some(graph),
        ..ModelProto::default()
    }
}

fn floats(values: &[f32]) -> TensorProto {
    TensorProto {
        data_type: Some(DataType::Float as i32),
        dims: vec![i64::try_from(values.len()).unwrap()],
        float_data: values.to_vec(),
        ..TensorProto::default()
    }
}

fn write(path: &Path, message: &impl Message) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, message.encode_to_vec()).unwrap();
}

/// A case of `model` with one data set per `(inputs, outputs)`.
fn case(
    root: &Path,
    name: &str,
    model: &ModelProto,
    sets: &[(Vec<TensorProto>, Vec<TensorProto>)],
) {
    let dir = root.join(name);
    write(&dir.join("model.onnx"), model);
    for (k, (inputs, outputs)) in sets.iter().enumerate() {
        let set = dir.join(format!("test_data_set_{k}"));
        inputs.iter().enumerate().for_each(|(i, t)| write(&set.join(format!("input_{i}.pb")), t));
        outputs.iter().enumerate().for_each(|(i, t)| write(&set.join(format!("output_{i}.pb")), t));
    }
}

fn relu_model() -> ModelProto {
    let ty = || tensor_type(DataType::Float, &[2]);
    model(vec![node("Relu", &["x"], &["y"])], vec![info("x", ty())], vec![info("y", ty())])
}

/// A `Relu` case with one data set per pair of input and expected output.
pub(crate) fn relu(root: &Path, name: &str, inputs: &[[f32; 2]], outputs: &[[f32; 2]]) {
    let sets: Vec<_> =
        inputs.iter().zip(outputs).map(|(x, y)| (vec![floats(x)], vec![floats(y)])).collect();
    case(root, name, &relu_model(), &sets);
}

fn cast_to_int64() -> AttributeProto {
    AttributeProto {
        name: Some("to".to_owned()),
        r#type: Some(AttributeType::Int as i32),
        i: Some(DataType::Int64 as i64),
        ..AttributeProto::default()
    }
}

/// Every outcome: one pass, seven failures and five unsupported tests.
pub(crate) fn write_all(root: &Path) {
    relu(root, "test_relu", &[[1.0, -2.0]], &[[1.0, 0.0]]);
    relu(root, "test_relu_wrong", &[[1.0, -2.0]], &[[1.0, 5.0]]);
    relu(root, "test_relu_second_set", &[[1.0, -2.0], [3.0, 4.0]], &[[1.0, 0.0], [3.0, 0.0]]);
    write_unsupported(root);
    write_failures(root);
}

/// Unsupported ops in the default and another domain, a sequence-typed
/// model whose data is not tensors,
/// unsupported test data, and a shape no input fixes.
fn write_unsupported(root: &Path) {
    let topk = model(
        vec![node("TopK", &["x", "k"], &["v", "i"])],
        vec![info("x", tensor_type(DataType::Float, &[2]))],
        vec![],
    );
    case(root, "test_topk", &topk, &[(vec![floats(&[1.0, 2.0])], vec![])]);
    let binarizer =
        NodeProto { domain: Some("ai.onnx.ml".to_owned()), ..node("Binarizer", &["x"], &["y"]) };
    let mut ml =
        model(vec![binarizer], vec![info("x", tensor_type(DataType::Float, &[2]))], vec![]);
    ml.opset_import
        .push(OperatorSetIdProto { domain: Some("ai.onnx.ml".to_owned()), version: Some(5) });
    case(root, "test_ml_domain", &ml, &[(vec![floats(&[1.0, 2.0])], vec![])]);
    let element = Some(Box::new(tensor_type(DataType::Float, &[2])));
    let sequence_type = TypeProto {
        value: Some(Value::SequenceType(Box::new(Sequence { elem_type: element }))),
        ..TypeProto::default()
    };
    let sequence =
        model(vec![node("Identity", &["s"], &["t"])], vec![info("s", sequence_type)], vec![]);
    case(root, "test_sequence", &sequence, &[]);
    std::fs::create_dir_all(root.join("test_sequence/test_data_set_0")).unwrap();
    std::fs::write(root.join("test_sequence/test_data_set_0/input_0.pb"), [0x0a, 0x03, 0x01, 0x02])
        .unwrap();
    let double = TensorProto {
        data_type: Some(DataType::Double as i32),
        dims: vec![2],
        double_data: vec![1.0, 0.0],
        ..TensorProto::default()
    };
    case(root, "test_double_data", &relu_model(), &[(vec![floats(&[1.0, -2.0])], vec![double])]);
    let cast = NodeProto { attribute: vec![cast_to_int64()], ..node("Cast", &["shape"], &["s"]) };
    let cast_shape = model(
        vec![cast, node("Reshape", &["x", "s"], &["y"])],
        vec![
            info("x", tensor_type(DataType::Float, &[2])),
            info("shape", tensor_type(DataType::Float, &[1])),
        ],
        vec![info("y", tensor_type(DataType::Float, &[2]))],
    );
    case(
        root,
        "test_cast_shape",
        &cast_shape,
        &[(vec![floats(&[1.0, 2.0]), floats(&[2.0])], vec![floats(&[1.0, 2.0])])],
    );
}

/// An import error, an evaluation error, an extra expected output, a
/// corrupt model, and unreadable data for a supported model.
fn write_failures(root: &Path) {
    let declared = model(
        vec![node("Relu", &["x"], &["y"])],
        vec![info("x", tensor_type(DataType::Float, &[2]))],
        vec![info("y", tensor_type(DataType::Float, &[3]))],
    );
    case(
        root,
        "test_declared",
        &declared,
        &[(vec![floats(&[1.0, 2.0])], vec![floats(&[1.0, 2.0])])],
    );
    let gather = model(
        vec![node("Gather", &["x", "i"], &["y"])],
        vec![
            info("x", tensor_type(DataType::Float, &[2])),
            info("i", tensor_type(DataType::Int64, &[1])),
        ],
        vec![info("y", tensor_type(DataType::Float, &[1]))],
    );
    let index = TensorProto {
        data_type: Some(DataType::Int64 as i32),
        dims: vec![1],
        int64_data: vec![5],
        ..TensorProto::default()
    };
    case(
        root,
        "test_gather_out_of_range",
        &gather,
        &[(vec![floats(&[1.0, 2.0]), index], vec![floats(&[1.0])])],
    );
    case(
        root,
        "test_extra_output",
        &relu_model(),
        &[(vec![floats(&[1.0, 2.0])], vec![floats(&[1.0, 2.0]), floats(&[0.0, 0.0])])],
    );
    std::fs::create_dir_all(root.join("test_corrupt")).unwrap();
    std::fs::write(root.join("test_corrupt/model.onnx"), [0xff, 0xff, 0xff]).unwrap();
    relu(root, "test_unreadable", &[[1.0, 2.0]], &[[1.0, 2.0]]);
    let unreadable = root.join("test_unreadable/test_data_set_0/input_0.pb");
    std::fs::remove_file(&unreadable).unwrap();
    std::fs::create_dir(&unreadable).unwrap();
}
