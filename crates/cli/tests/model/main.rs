//! `ptxc compile` and `ptxc run` end to end on a small MLP written as an
//! ONNX model, through the real binary. Running needs a GPU, so only the
//! failures before the device is touched are tested here.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

use prost::Message;
use ptxc_onnx_proto::tensor_proto::DataType;
use ptxc_onnx_proto::tensor_shape_proto::{Dimension, dimension};
use ptxc_onnx_proto::type_proto::{Tensor, Value};
use ptxc_onnx_proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto,
};
use std::path::Path;
use std::process::{Command, Output};

fn f32_info(name: &str, dims: &[i64]) -> ValueInfoProto {
    let dims: Vec<dimension::Value> = dims.iter().map(|&d| dimension::Value::DimValue(d)).collect();
    info(name, &dims)
}

/// A float tensor whose dimensions are each fixed or symbolic.
fn info(name: &str, dims: &[dimension::Value]) -> ValueInfoProto {
    let dim =
        dims.iter().map(|d| Dimension { value: Some(d.clone()), ..Dimension::default() }).collect();
    let tensor =
        Tensor { elem_type: Some(DataType::Float as i32), shape: Some(TensorShapeProto { dim }) };
    ValueInfoProto {
        name: Some(name.to_owned()),
        r#type: Some(TypeProto { value: Some(Value::TensorType(tensor)), ..TypeProto::default() }),
        ..ValueInfoProto::default()
    }
}

fn weights(name: &str, dims: &[i64]) -> TensorProto {
    let count = dims.iter().product::<i64>();
    let values =
        (0..count).map(|n| f32::from(u16::try_from(n % 7).unwrap()).mul_add(0.25, -0.5)).collect();
    TensorProto {
        name: Some(name.to_owned()),
        data_type: Some(DataType::Float as i32),
        dims: dims.to_vec(),
        float_data: values,
        ..TensorProto::default()
    }
}

fn node(op: &str, inputs: &[&str], output: &str, attribute: Vec<AttributeProto>) -> NodeProto {
    NodeProto {
        op_type: Some(op.to_owned()),
        input: inputs.iter().map(|&name| name.to_owned()).collect(),
        output: vec![output.to_owned()],
        attribute,
        ..NodeProto::default()
    }
}

/// `softmax(relu(x w1 + b1) w2 + b2)` for `x` of shape `[2, 4]`.
pub(crate) fn mlp() -> ModelProto {
    let axis = AttributeProto {
        name: Some("axis".into()),
        r#type: Some(ptxc_onnx_proto::attribute_proto::AttributeType::Int as i32),
        i: Some(-1),
        ..AttributeProto::default()
    };
    let graph = GraphProto {
        node: vec![
            node("MatMul", &["x", "w1"], "h", vec![]),
            node("Add", &["h", "b1"], "hb", vec![]),
            node("Relu", &["hb"], "r", vec![]),
            node("MatMul", &["r", "w2"], "o", vec![]),
            node("Add", &["o", "b2"], "ob", vec![]),
            node("Softmax", &["ob"], "y", vec![axis]),
        ],
        input: vec![f32_info("x", &[2, 4])],
        initializer: vec![
            weights("w1", &[4, 8]),
            weights("b1", &[8]),
            weights("w2", &[8, 3]),
            weights("b2", &[3]),
        ],
        output: vec![f32_info("y", &[2, 3])],
        ..GraphProto::default()
    };
    ModelProto {
        ir_version: Some(9),
        opset_import: vec![OperatorSetIdProto { domain: Some(String::new()), version: Some(21) }],
        graph: Some(graph),
        ..ModelProto::default()
    }
}

fn ptxc(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ptxc")).args(args).env_remove("RUST_LOG").output().unwrap()
}

fn write_model(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("mlp.onnx");
    std::fs::write(&path, mlp().encode_to_vec()).unwrap();
    path
}

#[test]
fn compile_writes_a_program_that_loads() {
    let dir = tempfile::tempdir().unwrap();
    let model = write_model(dir.path());
    let out = dir.path().join("mlp.ptxc");

    let output = ptxc(&["compile".as_ref(), model.as_os_str(), "-o".as_ref(), out.as_os_str()]);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(stderr.contains("x [2, 4]") && stderr.contains("y [2, 3]"), "{stderr}");
    let program = ptxc_program::load(&std::fs::read(&out).unwrap()).unwrap();
    assert!(program.kernels().len() >= 6, "one or more kernels per node");
    let names: Vec<&str> =
        program.inputs().iter().chain(program.outputs()).map(|b| b.name.as_str()).collect();
    assert_eq!(names, ["x", "y"]);
}

#[test]
fn a_missing_model_or_malformed_dim_fails_with_a_message() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("none.onnx");
    let out = dir.path().join("out.ptxc");

    let absent = ptxc(&["compile".as_ref(), missing.as_os_str(), "-o".as_ref(), out.as_os_str()]);
    let malformed = ptxc(&[
        "compile".as_ref(),
        missing.as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
        "--dim".as_ref(),
        "batch".as_ref(),
    ]);

    assert_eq!(absent.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&absent.stderr).contains("reading"),
        "{}",
        String::from_utf8_lossy(&absent.stderr)
    );
    assert_eq!(malformed.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&malformed.stderr).contains("NAME=SIZE"));
}

#[test]
fn run_reports_a_missing_input_file_before_using_the_gpu() {
    let dir = tempfile::tempdir().unwrap();
    let model = write_model(dir.path());
    let input = dir.path().join("absent.pb");

    let output = ptxc(&["run".as_ref(), model.as_os_str(), "--input".as_ref(), input.as_os_str()]);

    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("absent.pb"));
}

/// The MLP with its batch dimension named `batch`, and `w1` read from an
/// external data file next to the model.
fn batched_with_external_weights(dir: &Path) -> std::path::PathBuf {
    let mut model = mlp();
    let graph = model.graph.as_mut().unwrap();
    let batch = || vec![dimension::Value::DimParam("batch".into()), dimension::Value::DimValue(4)];
    graph.input = vec![info("x", &batch())];
    graph.output = vec![info(
        "y",
        &[dimension::Value::DimParam("batch".into()), dimension::Value::DimValue(3)],
    )];
    let w1 = &mut graph.initializer[0];
    let bytes: Vec<u8> = w1.float_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join("w1.bin"), bytes).unwrap();
    w1.float_data.clear();
    w1.data_location = Some(ptxc_onnx_proto::tensor_proto::DataLocation::External as i32);
    w1.external_data = vec![ptxc_onnx_proto::StringStringEntryProto {
        key: Some("location".into()),
        value: Some("w1.bin".into()),
    }];
    let path = dir.join("batched.onnx");
    std::fs::write(&path, model.encode_to_vec()).unwrap();
    path
}

#[test]
fn compile_sizes_symbolic_dims_and_reads_external_data_beside_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let model = batched_with_external_weights(dir.path());
    let out = dir.path().join("batched.ptxc");
    let elsewhere = tempfile::tempdir().unwrap();

    let sized = Command::new(env!("CARGO_BIN_EXE_ptxc"))
        .args([
            "compile".as_ref(),
            model.as_os_str(),
            "-o".as_ref(),
            out.as_os_str(),
            "--dim".as_ref(),
            "batch=6".as_ref(),
        ])
        .current_dir(elsewhere.path())
        .output()
        .unwrap();
    let unsized_run =
        ptxc(&["compile".as_ref(), model.as_os_str(), "-o".as_ref(), out.as_os_str()]);

    assert!(sized.status.success(), "{}", String::from_utf8_lossy(&sized.stderr));
    let program = ptxc_program::load(&std::fs::read(&out).unwrap()).unwrap();
    let x = &program.inputs().iter().next().unwrap().shape;
    assert_eq!(x.dims().iter().map(|d| d.get()).collect::<Vec<_>>(), [6, 4]);
    assert_eq!(unsized_run.status.code(), Some(2));
}

#[test]
fn fuzz_runs_its_seeds_on_the_interpreters_and_passes_when_they_agree() {
    let dir = tempfile::tempdir().unwrap();

    let output = ptxc(&[
        "fuzz".as_ref(),
        "--backend".as_ref(),
        "interp".as_ref(),
        "--cases".as_ref(),
        "40".as_ref(),
        "--out".as_ref(),
        dir.path().as_os_str(),
    ]);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(stderr.contains("agreed") && stderr.contains("40"), "{stderr}");
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}
