//! Quantize, dequantize, `QLinearMatMul` and `QLinearConv`, with scales and
//! zero points as attributes.

use crate::builders::{
    FLOAT, assert_lines, f32_tensor, fixed, import, input_names, int, line, model, node, op_line,
    typed,
};
use ptxc_onnx::Unsupported;
use ptxc_onnx_proto::tensor_proto::DataType;
use ptxc_onnx_proto::{AttributeProto, TensorProto};

fn quantize(params: Vec<TensorProto>, attrs: Vec<AttributeProto>) -> String {
    let names = input_names("x", &params);
    line(node("QuantizeLinear", &names, &["y"], attrs), vec![fixed("x", FLOAT, &[2, 3])], params)
}

fn dequantize(dtype: DataType, params: Vec<TensorProto>) -> String {
    let names = input_names("x", &params);
    line(node("DequantizeLinear", &names, &["y"], vec![]), vec![fixed("x", dtype, &[3])], params)
}

#[test]
fn scale_and_zero_point_types_set_the_parameters() {
    let half = typed("s", DataType::Float16, &[], &[0x3800]);
    let brain = typed("s", DataType::Bfloat16, &[], &[0x3f00]);

    assert_lines(&[
        (
            quantize(vec![f32_tensor("s", &[], &[0.5])], vec![]),
            "quantize<params=[u8, [0.5, 0]]> %x : u8[2, 3]",
        ),
        (
            quantize(
                vec![f32_tensor("s", &[], &[0.5]), typed("z", DataType::Int8, &[], &[-3])],
                vec![],
            ),
            "quantize<params=[i8, [0.5, -3]]> %x : i8[2, 3]",
        ),
        (
            quantize(
                vec![half, typed("z", DataType::Float8e4m3fn, &[], &[0])],
                vec![int("saturate", 0)],
            ),
            "quantize<params=[f8e4m3fn, [0.5, 0]], overflow=non_finite> %x : f8e4m3fn[2, 3]",
        ),
        (
            quantize(vec![brain, typed("z", DataType::Float8e5m2, &[], &[0])], vec![]),
            "quantize<params=[f8e5m2, [0.5, 0]], overflow=saturate> %x : f8e5m2[2, 3]",
        ),
        (
            quantize(vec![f32_tensor("s", &[3], &[0.5, 0.25, 2.0])], vec![int("output_dtype", 3)]),
            "quantize<params=[i8, 1, [[0.5, 0], [0.25, 0], [2.0, 0]]]> %x : i8[2, 3]",
        ),
        (
            dequantize(
                DataType::Int32,
                vec![f32_tensor("s", &[], &[0.5]), typed("z", DataType::Int32, &[], &[7])],
            ),
            "dequantize<params=[i32, [0.5, 7]], output=f32> %x : f32[3]",
        ),
    ]);
}

#[test]
fn nonzero_fp8_zero_points_and_blocked_scales_are_unsupported() {
    let fp8 = model(
        vec![node("QuantizeLinear", &["x", "s", "z"], &["y"], vec![])],
        vec![fixed("x", FLOAT, &[2])],
        vec![f32_tensor("s", &[], &[0.5]), typed("z", DataType::Float8e4m3fn, &[], &[0x38])],
        vec![],
    );
    let blocked = model(
        vec![node("QuantizeLinear", &["x", "s"], &["y"], vec![int("block_size", 2)])],
        vec![fixed("x", FLOAT, &[2, 4])],
        vec![f32_tensor("s", &[2, 2], &[1.0; 4])],
        vec![],
    );

    assert_eq!(
        import(&fp8).unwrap_err().unsupported(),
        Some(&Unsupported::Attribute {
            op: "QuantizeLinear".to_owned(),
            attribute: "nonzero FP8 zero point".to_owned()
        })
    );
    assert_eq!(
        import(&blocked).unwrap_err().unsupported(),
        Some(&Unsupported::Attribute {
            op: "QuantizeLinear".to_owned(),
            attribute: "block_size".to_owned()
        })
    );
}

#[test]
fn qlinear_matmul_takes_rows_of_a_and_columns_of_b() {
    let u8s = |name: &str, dims: &[i64], values: &[i32]| typed(name, DataType::Uint8, dims, values);
    let constants = vec![
        f32_tensor("as", &[2], &[0.5, 0.25]),
        u8s("az", &[2], &[1, 2]),
        f32_tensor("bs", &[4], &[1.0, 2.0, 3.0, 4.0]),
        u8s("bz", &[4], &[0, 0, 0, 0]),
        f32_tensor("ys", &[], &[0.125]),
        u8s("yz", &[], &[9]),
    ];
    let inputs = vec![fixed("a", DataType::Uint8, &[2, 3]), fixed("b", DataType::Uint8, &[3, 4])];
    let names = ["a", "as", "az", "b", "bs", "bz", "ys", "yz"];

    assert_lines(&[(
        line(node("QLinearMatMul", &names, &["y"], vec![]), inputs, constants),
        "qmatmul<a=[u8, 0, [[0.5, 1], [0.25, 2]]], b=[u8, 1, [[1.0, 0], [2.0, 0], [3.0, 0], [4.0, 0]]], output=[u8, [0.125, 9]]> %a, %b : u8[2, 4]",
    )]);
}

#[test]
fn qlinear_conv_takes_weight_scales_per_output_channel() {
    let constants = vec![
        f32_tensor("xs", &[], &[0.5]),
        typed("xz", DataType::Uint8, &[], &[3]),
        typed("w", DataType::Int8, &[2, 1, 3], &[0; 6]),
        f32_tensor("ws", &[2], &[0.25, 0.5]),
        typed("wz", DataType::Int8, &[2], &[0, 0]),
        f32_tensor("ys", &[], &[1.0]),
        typed("yz", DataType::Uint8, &[], &[0]),
        typed("b", DataType::Int32, &[2], &[5, 6]),
    ];
    let names = ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz", "b"];

    let conv = line(
        node("QLinearConv", &names, &["y"], vec![]),
        vec![fixed("x", DataType::Uint8, &[1, 1, 5])],
        constants,
    );

    assert_lines(&[(
        conv,
        "qconv<groups=1, axes=[[1, 1, 0, 0]], input=[u8, [0.5, 3]], weights=[i8, 0, [[0.25, 0], [0.5, 0]]], output=[u8, [1.0, 0]]> %x, %w, %b : u8[1, 2, 3]",
    )]);
}

#[test]
fn a_negative_quantization_axis_counts_from_the_input_rank() {
    let scales = f32_tensor("s", &[3], &[0.5, 0.25, 2.0]);
    let node = node("QuantizeLinear", &["x", "s"], &["y"], vec![int("axis", -1)]);

    let line = op_line(&model(vec![node], vec![fixed("x", FLOAT, &[2, 3])], vec![scales], vec![]));

    assert!(
        line.starts_with("quantize<params=[u8, 1, [[0.5, 0], [0.25, 0], [2.0, 0]]]>"),
        "{line}"
    );
}

#[test]
fn a_one_element_scale_is_per_tensor_even_where_the_default_axis_does_not_exist() {
    let scale = f32_tensor("s", &[1], &[0.5]);
    let node = node("DequantizeLinear", &["x", "s"], &["y"], vec![]);
    let x = fixed("x", DataType::Int8, &[4]);

    let line = op_line(&model(vec![node], vec![x], vec![scale], vec![]));

    assert!(line.starts_with("dequantize<params=[i8, [0.5, 0]], output=f32>"), "{line}");
}

#[test]
fn batched_qlinear_matmul_takes_columns_of_bs_last_axis() {
    let u8s = |name: &str, dims: &[i64], values: &[i32]| typed(name, DataType::Uint8, dims, values);
    let constants = vec![
        f32_tensor("as", &[], &[0.5]),
        u8s("az", &[], &[0]),
        f32_tensor("bs", &[4], &[1.0, 2.0, 3.0, 4.0]),
        u8s("bz", &[4], &[0, 0, 0, 0]),
        f32_tensor("ys", &[], &[1.0]),
        u8s("yz", &[], &[0]),
    ];
    let inputs =
        vec![fixed("a", DataType::Uint8, &[2, 2, 3]), fixed("b", DataType::Uint8, &[2, 3, 4])];
    let names = ["a", "as", "az", "b", "bs", "bz", "ys", "yz"];

    assert_lines(&[(
        line(node("QLinearMatMul", &names, &["y"], vec![]), inputs, constants),
        "qmatmul<a=[u8, [0.5, 0]], b=[u8, 2, [[1.0, 0], [2.0, 0], [3.0, 0], [4.0, 0]]], output=[u8, [1.0, 0]]> %a, %b : u8[2, 2, 4]",
    )]);
}
