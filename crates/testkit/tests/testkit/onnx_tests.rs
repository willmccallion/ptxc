//! Loading test cases written to a temporary directory.

use prost::Message;
use ptxc_core::TensorData;
use ptxc_onnx::Unsupported;
use ptxc_onnx_proto::tensor_proto::DataType;
use ptxc_onnx_proto::tensor_shape_proto::{Dimension, dimension};
use ptxc_onnx_proto::type_proto::{Tensor, Value};
use ptxc_onnx_proto::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto,
    TypeProto, ValueInfoProto,
};
use ptxc_testkit::onnx_tests::{LoadError, TestCase, cases};
use std::path::Path;

fn info(name: &str, dtype: DataType, dims: &[i64]) -> ValueInfoProto {
    let dim = dims
        .iter()
        .map(|&d| Dimension { value: Some(dimension::Value::DimValue(d)), ..Dimension::default() })
        .collect();
    let tensor = Tensor { elem_type: Some(dtype as i32), shape: Some(TensorShapeProto { dim }) };
    ValueInfoProto {
        name: Some(name.to_owned()),
        r#type: Some(TypeProto { value: Some(Value::TensorType(tensor)), ..TypeProto::default() }),
        ..ValueInfoProto::default()
    }
}

/// `y = Reshape(x, shape) + bias`, with the shape a graph input and the
/// bias an initializer the old way, listed among the inputs too.
fn reshape_model() -> ModelProto {
    let nodes = vec![
        NodeProto {
            op_type: Some("Reshape".to_owned()),
            input: vec!["x".to_owned(), "shape".to_owned()],
            output: vec!["r".to_owned()],
            ..NodeProto::default()
        },
        NodeProto {
            op_type: Some("Add".to_owned()),
            input: vec!["r".to_owned(), "bias".to_owned()],
            output: vec!["y".to_owned()],
            ..NodeProto::default()
        },
    ];
    let bias = TensorProto {
        name: Some("bias".to_owned()),
        data_type: Some(DataType::Float as i32),
        dims: vec![1],
        float_data: vec![0.5],
        ..TensorProto::default()
    };
    let graph = GraphProto {
        node: nodes,
        input: vec![
            info("x", DataType::Float, &[2, 2]),
            info("bias", DataType::Float, &[1]),
            info("shape", DataType::Int64, &[1]),
        ],
        initializer: vec![bias],
        output: vec![info("y", DataType::Float, &[4])],
        ..GraphProto::default()
    };
    ModelProto {
        ir_version: Some(9),
        opset_import: vec![OperatorSetIdProto { domain: Some(String::new()), version: Some(21) }],
        graph: Some(graph),
        ..ModelProto::default()
    }
}

fn floats(values: &[f32], dims: &[i64]) -> TensorProto {
    TensorProto {
        data_type: Some(DataType::Float as i32),
        dims: dims.to_vec(),
        float_data: values.to_vec(),
        ..TensorProto::default()
    }
}

fn write(path: &Path, message: &impl Message) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, message.encode_to_vec()).unwrap();
}

/// A case directory holding the model and data sets of `x` and `shape`.
fn write_case(root: &Path, name: &str, sets: &[[f32; 4]]) -> std::path::PathBuf {
    let dir = root.join(name);
    write(&dir.join("model.onnx"), &reshape_model());
    for (k, x) in sets.iter().enumerate() {
        let set = dir.join(format!("test_data_set_{k}"));
        write(&set.join("input_0.pb"), &floats(x, &[2, 2]));
        write(
            &set.join("input_1.pb"),
            &TensorProto {
                data_type: Some(DataType::Int64 as i32),
                dims: vec![1],
                int64_data: vec![4],
                ..TensorProto::default()
            },
        );
        write(&set.join("output_0.pb"), &floats(&x.map(|v| v + 0.5), &[4]));
    }
    dir
}

#[test]
fn cases_load_in_name_order_with_every_data_set() {
    let root = tempfile::tempdir().unwrap();
    let _later = write_case(root.path(), "test_b", &[[1.0, 2.0, 3.0, 4.0]]);
    let _earlier = write_case(root.path(), "test_a", &[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);

    let loaded = cases(root.path()).unwrap();

    let names: Vec<&str> = loaded.iter().map(TestCase::name).collect();
    assert_eq!(names, ["test_a", "test_b"]);
    let sets = loaded[0].data_sets().unwrap();
    assert_eq!(sets.len(), 2);
    assert_eq!((sets[1].inputs.len(), sets[1].outputs.len()), (2, 1));
    assert!(
        matches!(sets[1].inputs[0].data(), TensorData::F32(values) if values == &[5.0, 6.0, 7.0, 8.0])
    );
    assert_eq!(loaded[0].model(), reshape_model().encode_to_vec().as_slice());
}

#[test]
fn importing_binds_the_inputs_a_shape_needs_and_keeps_the_rest() {
    let root = tempfile::tempdir().unwrap();
    let case =
        TestCase::load(&write_case(root.path(), "test_reshape", &[[1.0, 2.0, 3.0, 4.0]])).unwrap();
    let set = case.data_sets().unwrap().remove(0);

    let runnable = case.import(&set).unwrap();

    assert_eq!(runnable.graph.inputs().iter().count(), 1);
    assert_eq!(runnable.inputs.len(), 1);
    assert!(
        matches!(runnable.inputs[0].data(), TensorData::F32(values) if values == &[1.0, 2.0, 3.0, 4.0])
    );
}

#[test]
fn a_shape_that_binding_cannot_make_known_is_left_to_the_caller() {
    let root = tempfile::tempdir().unwrap();
    let dir = write_case(root.path(), "test_cast", &[[1.0, 2.0, 3.0, 4.0]]);
    let mut model = reshape_model();
    let graph = model.graph.as_mut().unwrap();
    graph.input[2] = info("shape", DataType::Float, &[1]);
    let to_int = ptxc_onnx_proto::AttributeProto {
        name: Some("to".to_owned()),
        r#type: Some(ptxc_onnx_proto::attribute_proto::AttributeType::Int as i32),
        i: Some(DataType::Int64 as i64),
        ..ptxc_onnx_proto::AttributeProto::default()
    };
    let cast = NodeProto {
        op_type: Some("Cast".to_owned()),
        input: vec!["shape".to_owned()],
        output: vec!["s".to_owned()],
        attribute: vec![to_int],
        ..NodeProto::default()
    };
    graph.node[0].input[1] = "s".to_owned();
    graph.node.insert(0, cast);
    write(&dir.join("model.onnx"), &model);
    write(&dir.join("test_data_set_0/input_1.pb"), &floats(&[4.0], &[1]));
    let case = TestCase::load(&dir).unwrap();

    let error = case.import(&case.data_sets().unwrap()[0]).unwrap_err();

    let ptxc_onnx::ImportErrorKind::NeedsConstant { inputs, .. } = error.kind() else {
        panic!("{error}")
    };
    assert_eq!(inputs, &["shape".to_owned()]);
}

#[test]
fn unsupported_test_data_is_a_typed_tensor_error() {
    let root = tempfile::tempdir().unwrap();
    let dir = write_case(root.path(), "test_double", &[[1.0, 2.0, 3.0, 4.0]]);
    let double = TensorProto {
        data_type: Some(DataType::Double as i32),
        dims: vec![1],
        double_data: vec![1.0],
        ..TensorProto::default()
    };
    write(&dir.join("test_data_set_0/output_0.pb"), &double);

    let error = TestCase::load(&dir).unwrap().data_sets().unwrap_err();

    let LoadError::Tensor { path, source } = error else { panic!("{error}") };
    assert!(path.ends_with("test_data_set_0/output_0.pb"));
    assert_eq!(source.unsupported(), Some(&Unsupported::DType("DOUBLE".to_owned())));
}

#[test]
fn a_missing_or_corrupt_model_is_a_load_error() {
    let root = tempfile::tempdir().unwrap();
    let corrupt = root.path().join("test_corrupt");
    std::fs::create_dir_all(&corrupt).unwrap();
    std::fs::write(corrupt.join("model.onnx"), [0xff, 0xff, 0xff]).unwrap();

    assert!(matches!(TestCase::load(&root.path().join("test_absent")), Err(LoadError::Io { .. })));
    assert!(matches!(TestCase::load(&corrupt), Err(LoadError::Model { .. })));
}

#[test]
fn an_unreadable_tensor_file_is_an_io_error() {
    let root = tempfile::tempdir().unwrap();
    let dir = write_case(root.path(), "test_unreadable", &[[1.0, 2.0, 3.0, 4.0]]);
    std::fs::remove_file(dir.join("test_data_set_0/input_1.pb")).unwrap();
    std::fs::create_dir(dir.join("test_data_set_0/input_1.pb")).unwrap();

    let error = TestCase::load(&dir).unwrap().data_sets().unwrap_err();

    assert!(matches!(error, LoadError::Io { .. }), "{error}");
}
