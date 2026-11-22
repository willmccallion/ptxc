//! The generated types decode protobuf bytes written by hand from the wire
//! format, so field numbers, wire types and proto2 presence match onnx.proto.

#![allow(clippy::unwrap_used, clippy::float_cmp, reason = "test code: exact values on purpose")]

use prost::Message;
use ptxc_onnx_proto::TensorProto;
use ptxc_onnx_proto::tensor_proto::DataType;

#[test]
fn tensor_fields_decode_from_their_onnx_field_numbers() {
    let bytes = [
        0x08, 2, // field 1 (dims), varint 2
        0x08, 3, // field 1 (dims), varint 3: repeated, unpacked
        0x10, 1, // field 2 (data_type), varint 1
        0x42, 1, b'w', // field 8 (name), 1 byte
        0x4A, 2, 0xAB, 0xCD, // field 9 (raw_data), 2 bytes
    ];

    let tensor = TensorProto::decode(bytes.as_slice()).unwrap();

    assert_eq!(tensor.dims, [2, 3]);
    assert_eq!(tensor.data_type, Some(1));
    assert_eq!(tensor.name.as_deref(), Some("w"));
    assert_eq!(tensor.raw_data.as_deref(), Some([0xAB, 0xCD].as_slice()));
}

#[test]
fn float_data_decodes_from_field_4_as_onnx_writes_it() {
    let mut bytes = vec![0x22, 8]; // field 4 (float_data), 8 bytes
    bytes.extend(1.0_f32.to_le_bytes());
    bytes.extend((-2.0_f32).to_le_bytes());

    let tensor = TensorProto::decode(bytes.as_slice()).unwrap();

    assert_eq!(tensor.float_data, [1.0, -2.0]);
}

#[test]
fn absent_and_zero_data_type_are_distinguished() {
    let absent = TensorProto::decode([].as_slice()).unwrap();
    let zero = TensorProto::decode([0x10, 0].as_slice()).unwrap();

    assert_eq!((absent.data_type, zero.data_type), (None, Some(0)));
}

#[test]
fn data_type_codes_match_onnx() {
    let codes = [
        (DataType::Float, 1),
        (DataType::Uint8, 2),
        (DataType::Int8, 3),
        (DataType::Int32, 6),
        (DataType::Int64, 7),
        (DataType::Bool, 9),
        (DataType::Float16, 10),
        (DataType::Bfloat16, 16),
        (DataType::Float8e4m3fn, 17),
        (DataType::Float8e5m2, 19),
    ];

    for (data_type, code) in codes {
        assert_eq!(data_type as i32, code, "{data_type:?}");
    }
}
