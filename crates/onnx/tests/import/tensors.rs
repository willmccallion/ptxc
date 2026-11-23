//! Tensor import: raw bytes, typed fields, external data and their errors.

use crate::builders::{FLOAT, fixed, import_with, model, node};
use prost::Message;
use ptxc_core::{DType, TensorData};
use ptxc_onnx::{ImportErrorKind, ImportOptions, Unsupported, tensor_from_bytes};
use ptxc_onnx_proto::{StringStringEntryProto, TensorProto};

fn proto(data_type: i32, dims: &[i64]) -> TensorProto {
    TensorProto {
        name: Some("t".to_owned()),
        data_type: Some(data_type),
        dims: dims.to_vec(),
        ..TensorProto::default()
    }
}

fn decode(tensor: &TensorProto) -> Result<ptxc_core::HostTensor, ptxc_onnx::ImportError> {
    tensor_from_bytes(&tensor.encode_to_vec())
}

#[test]
fn raw_data_is_little_endian() {
    let tensor = TensorProto {
        raw_data: Some([1.5_f32, -2.0].iter().flat_map(|v| v.to_le_bytes()).collect()),
        ..proto(1, &[2])
    };

    let host = decode(&tensor).unwrap();

    assert!(matches!(host.data(), TensorData::F32(values) if *values == [1.5, -2.0]));
}

#[test]
fn narrow_types_read_their_bits_and_values_from_int32_data() {
    let half = TensorProto { int32_data: vec![0x3c00], ..proto(10, &[1]) };
    let flags = TensorProto { int32_data: vec![1, 0], ..proto(9, &[2]) };

    assert!(
        matches!(decode(&half).unwrap().data(), TensorData::F16(values) if values[0].to_bits() == 0x3c00)
    );
    assert!(
        matches!(decode(&flags).unwrap().data(), TensorData::Bool(values) if *values == [true, false])
    );
}

#[test]
fn raw_bools_are_true_when_nonzero() {
    let tensor = TensorProto { raw_data: Some(vec![0, 2, 1]), ..proto(9, &[3]) };

    assert!(
        matches!(decode(&tensor).unwrap().data(), TensorData::Bool(values) if *values == [false, true, true])
    );
}

#[test]
fn values_that_do_not_fit_are_invalid() {
    let tensor = TensorProto { int32_data: vec![300], ..proto(3, &[1]) };

    assert!(matches!(decode(&tensor).unwrap_err().kind(), ImportErrorKind::Invalid { .. }));
}

#[test]
fn unsupported_element_types_are_named() {
    let error = decode(&TensorProto { double_data: vec![1.0], ..proto(11, &[1]) }).unwrap_err();

    assert_eq!(error.unsupported(), Some(&Unsupported::DType("DOUBLE".to_owned())));
}

#[test]
fn data_that_does_not_fill_the_dims_is_an_error() {
    let error = decode(&TensorProto { float_data: vec![1.0, 2.0], ..proto(1, &[3]) }).unwrap_err();

    assert!(
        matches!(error.kind(), ImportErrorKind::DataLength { expected: 3, actual: 2, .. }),
        "{error}"
    );
}

fn external(location: &str, offset: Option<&str>, length: Option<&str>) -> TensorProto {
    let entry = |key: &str, value: &str| StringStringEntryProto {
        key: Some(key.to_owned()),
        value: Some(value.to_owned()),
    };
    let mut entries = vec![entry("location", location)];
    entries.extend(offset.map(|value| entry("offset", value)));
    entries.extend(length.map(|value| entry("length", value)));
    TensorProto {
        name: Some("w".to_owned()),
        data_location: Some(1),
        external_data: entries,
        ..proto(1, &[2])
    }
}

fn import_external(
    tensor: TensorProto,
    dir: &std::path::Path,
) -> Result<ptxc_graph::Graph, ptxc_onnx::ImportError> {
    let model = model(
        vec![node("Identity", &["w"], &["y"], vec![])],
        vec![],
        vec![tensor],
        vec![fixed("y", FLOAT, &[2])],
    );
    import_with(
        &model,
        &ImportOptions { model_dir: Some(dir.to_owned()), ..ImportOptions::default() },
    )
}

#[test]
fn external_data_is_read_from_the_offset_and_length_given() {
    let dir = tempfile::tempdir().unwrap();
    let bytes: Vec<u8> = [9.0_f32, 1.0, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.path().join("weights.bin"), bytes).unwrap();

    let graph = import_external(external("weights.bin", Some("4"), Some("8")), dir.path()).unwrap();

    let output = graph.outputs().iter().copied().next().unwrap();
    let ptxc_graph::ValueDef::Constant(id) = graph.value(output).def() else {
        panic!("not a constant")
    };
    assert!(matches!(graph.constant(id).data(), TensorData::F32(values) if *values == [1.0, 2.0]));
    assert_eq!(graph.constant(id).dtype(), DType::F32);
}

#[test]
fn external_data_may_not_leave_the_model_directory() {
    let dir = tempfile::tempdir().unwrap();

    let escaping = import_external(external("../secret.bin", None, None), dir.path()).unwrap_err();
    let missing = import_external(external("absent.bin", None, None), dir.path()).unwrap_err();

    assert!(
        matches!(escaping.kind(), ImportErrorKind::Invalid { field: "external data location", .. }),
        "{escaping}"
    );
    assert!(matches!(missing.kind(), ImportErrorKind::ExternalData { .. }), "{missing}");
}

#[test]
fn external_data_past_the_end_of_the_file_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("short.bin"), [0_u8; 4]).unwrap();

    let error =
        import_external(external("short.bin", Some("0"), Some("8")), dir.path()).unwrap_err();

    assert!(
        matches!(error.kind(), ImportErrorKind::Invalid { field: "external data", .. }),
        "{error}"
    );
}
