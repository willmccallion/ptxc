//! Quantization as ONNX defines it, on values worked out by hand.

use crate::builders::{assert_close, assert_exact, f32s, op, tensor, values};
use ptxc_core::float::F8E4M3Fn;

#[test]
fn quantize_rounds_half_to_even_adds_the_zero_point_and_saturates() {
    let x = f32s(&[4], &[-1.0, 0.25, 0.75, 200.0]);

    assert_exact(&op("quantize<params=[u8, [0.5, 3]]>", "u8[4]", &[x]), &[4], &[1, 3, 5, 255]);
}

#[test]
fn per_axis_quantization_uses_each_index_map() {
    let x = f32s(&[2, 2], &[1.0, 1.0, -1.0, 300.0]);

    let quantized = op("quantize<params=[i8, 1, [[0.5, 0], [0.25, 1]]]>", "i8[2, 2]", &[x]);

    assert_exact(&quantized, &[2, 2], &[2, 5, -2, 127]);
}

#[test]
fn fp8_quantization_casts_the_quotient_as_saturate_says() {
    let x = || f32s(&[2], &[3.0, 2000.0]);

    let saturated =
        op("quantize<params=[f8e4m3fn, [2.0, 0]], overflow=saturate>", "f8e4m3fn[2]", &[x()]);
    let non_finite =
        op("quantize<params=[f8e4m3fn, [2.0, 0]], overflow=non_finite>", "f8e4m3fn[2]", &[x()]);

    assert_close(&saturated, &[2], &[1.5, 448.0]);
    assert!(values(&non_finite)[1].is_nan());
}

#[test]
fn dequantize_subtracts_the_zero_point_exactly_then_scales() {
    let bytes = tensor(&[3], vec![0_u8, 3, 255]);
    let wide = tensor(&[1], vec![16_777_217_i32]);
    let fp8 = tensor(&[1], vec![F8E4M3Fn::from_f32(1.5, ptxc_core::float::Overflow::Saturate)]);

    assert_close(
        &op("dequantize<params=[u8, [0.5, 3]], output=f32>", "f32[3]", &[bytes]),
        &[3],
        &[-1.5, 0.0, 126.0],
    );
    assert_close(
        &op("dequantize<params=[i32, [1.0, 1]], output=f32>", "f32[1]", &[wide]),
        &[1],
        &[16_777_216.0],
    );
    assert_close(
        &op("dequantize<params=[f8e4m3fn, [2.0, 0]], output=f16>", "f16[1]", &[fp8]),
        &[1],
        &[3.0],
    );
    let per_axis = op(
        "dequantize<params=[u8, 0, [[1.0, 0], [2.0, 1]]], output=f32>",
        "f32[2]",
        &[tensor(&[2], vec![5_u8, 5])],
    );
    assert_close(&per_axis, &[2], &[5.0, 8.0]);
}

#[test]
fn qlinear_matmul_rescales_exact_sums_per_row_and_column() {
    let a = tensor(&[2, 2], vec![4_u8, 5, 6, 7]);
    let b = tensor(&[2, 2], vec![1_u8, 2, 3, 4]);
    let attrs = "qmatmul<a=[u8, 0, [[0.5, 4], [1.0, 6]]], b=[u8, 1, [[1.0, 1], [2.0, 2]]], output=[u8, [0.25, 10]]>";

    assert_exact(&op(attrs, "u8[2, 2]", &[a, b]), &[2, 2], &[14, 18, 18, 26]);
}

#[test]
fn qlinear_matmul_rounds_half_to_even_before_the_zero_point() {
    let a = tensor(&[1, 1], vec![3_u8]);
    let b = tensor(&[1, 2], vec![1_u8, 5]);
    let attrs = "qmatmul<a=[u8, [1.0, 0]], b=[u8, [1.0, 0]], output=[u8, [2.0, 1]]>";

    assert_exact(&op(attrs, "u8[1, 2]", &[a, b]), &[1, 2], &[3, 9]);
}

#[test]
fn qlinear_conv_adds_the_bias_and_rescales_per_output_channel() {
    let x = tensor(&[1, 1, 3], vec![3_u8, 5, 7]);
    let w = tensor(&[2, 1, 2], vec![1_i8, 1, 2, -1]);
    let bias = tensor(&[2], vec![1_i32, -2]);
    let attrs = "qconv<groups=1, axes=[[1, 1, 0, 0]], input=[u8, [0.5, 3]], weights=[i8, 0, [[1.0, 0], [0.5, 0]]], output=[i8, [1.0, 0]]>";

    assert_exact(&op(attrs, "i8[1, 2, 2]", &[x, w, bias]), &[1, 2, 2], &[2, 4, -1, 0]);
}

#[test]
fn qlinear_matmul_sums_every_inner_product() {
    let a = tensor(&[1, 2], vec![2_u8, 3]);
    let b = tensor(&[2, 1], vec![4_u8, 5]);
    let attrs = "qmatmul<a=[u8, [1.0, 0]], b=[u8, [1.0, 0]], output=[u8, [1.0, 0]]>";

    assert_exact(&op(attrs, "u8[1, 1]", &[a, b]), &[1, 1], &[23]);
}

#[test]
fn qlinear_conv_without_a_bias() {
    let x = tensor(&[1, 1, 3], vec![3_u8, 5, 7]);
    let w = tensor(&[2, 1, 2], vec![1_i8, 1, 2, -1]);
    let attrs = "qconv<groups=1, axes=[[1, 1, 0, 0]], input=[u8, [0.5, 3]], weights=[i8, 0, [[1.0, 0], [0.5, 0]]], output=[i8, [1.0, 0]]>";

    assert_exact(&op(attrs, "i8[1, 2, 2]", &[x, w]), &[1, 2, 2], &[1, 3, 0, 0]);
}

#[test]
fn a_dequantized_fp8_matmul_scales_the_exact_sum_per_column() {
    let e4m3 = |values: &[f32]| {
        let overflow = ptxc_core::float::Overflow::Saturate;
        values.iter().map(|&value| F8E4M3Fn::from_f32(value, overflow)).collect::<Vec<_>>()
    };
    let a = tensor(&[2, 2], e4m3(&[1.0, 2.0, 3.0, 4.0]));
    let b = tensor(&[2, 2], e4m3(&[1.0, 0.0, 0.0, 2.0]));
    let attrs =
        "qmatmul<a=[f8e4m3fn, [0.5, 0]], b=[f8e4m3fn, 1, [[1.0, 0], [0.25, 0]]], result=f32>";

    // a @ b = [[1, 4], [3, 8]], times 0.5 and each column's scale.
    assert_close(&op(attrs, "f32[2, 2]", &[a, b]), &[2, 2], &[0.5, 0.5, 1.5, 1.0]);
}

#[test]
fn a_dequantized_integer_matmul_scales_the_exact_zero_centred_sum() {
    let a = tensor(&[2, 2], vec![4_u8, 5, 6, 7]);
    let b = tensor(&[2, 2], vec![1_u8, 2, 3, 4]);
    let attrs =
        "qmatmul<a=[u8, 0, [[0.5, 4], [1.0, 6]]], b=[u8, 1, [[1.0, 1], [2.0, 2]]], result=f16>";

    // Each zero-centred sum is 2, times the row's and the column's scales.
    assert_close(&op(attrs, "f16[2, 2]", &[a, b]), &[2, 2], &[1.0, 2.0, 2.0, 4.0]);
}
