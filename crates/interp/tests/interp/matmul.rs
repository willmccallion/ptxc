//! `MatMul` and `Gemm`, on products small enough to check by hand.

use crate::builders::{assert_close, assert_exact, f32s, i64s, op, tensor};

/// `[[1, 2, 3], [4, 5, 6]]`.
fn a() -> ptxc_core::HostTensor {
    f32s(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

/// `[[1, 0], [0, 1], [1, 1]]`.
fn b() -> ptxc_core::HostTensor {
    f32s(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
}

#[test]
fn matrices_multiply_rows_by_columns() {
    assert_close(&op("matmul", "f32[2, 2]", &[a(), b()]), &[2, 2], &[4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn batch_axes_broadcast() {
    let batched = f32s(&[2, 1, 3], &[1.0, 2.0, 3.0, -1.0, 0.0, 2.0]);

    let product = op("matmul", "f32[2, 1, 2]", &[batched, b()]);

    assert_close(&product, &[2, 1, 2], &[4.0, 5.0, 1.0, 2.0]);
    let left = f32s(&[2, 1, 1, 1], &[2.0, 3.0]);
    let right = f32s(&[1, 3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let both = op("matmul", "f32[2, 3, 1, 2]", &[left, right]);
    assert_close(
        &both,
        &[2, 3, 1, 2],
        &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
    );
}

#[test]
fn vectors_are_promoted_and_the_added_axis_removed() {
    let vector = f32s(&[3], &[1.0, -1.0, 2.0]);

    assert_close(&op("matmul", "f32[2]", &[vector.clone(), b()]), &[2], &[3.0, 1.0]);
    assert_close(&op("matmul", "f32[2]", &[a(), vector.clone()]), &[2], &[5.0, 11.0]);
    assert_close(&op("matmul", "f32[]", &[vector.clone(), vector]), &[], &[6.0]);
}

#[test]
fn integer_products_wrap() {
    let big = tensor(&[1, 2], vec![100_000_i32, 100_000]);
    let ones = tensor(&[2, 1], vec![100_000_i32, 1]);

    assert_exact(&op("matmul", "i32[1, 1]", &[big, ones]), &[1, 1], &[1_410_165_408]);
    assert_exact(
        &op("matmul", "i64[1, 1]", &[i64s(&[1, 2], &[3, 4]), i64s(&[2, 1], &[5, 6])]),
        &[1, 1],
        &[39],
    );
}

#[test]
fn gemm_orients_scales_and_adds_a_broadcast_bias() {
    let a_t = f32s(&[3, 2], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let b_t = f32s(&[2, 3], &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
    let row_bias = f32s(&[2], &[10.0, 20.0]);
    let column_bias = f32s(&[2, 1], &[100.0, 200.0]);

    let plain = op("gemm<a=as_is, b=as_is, alpha=1.0, beta=1.0>", "f32[2, 2]", &[a(), b()]);
    let transposed = op(
        "gemm<a=transposed, b=transposed, alpha=2.0, beta=0.5>",
        "f32[2, 2]",
        &[a_t, b_t, row_bias],
    );
    let columns =
        op("gemm<a=as_is, b=as_is, alpha=1.0, beta=1.0>", "f32[2, 2]", &[a(), b(), column_bias]);

    assert_close(&plain, &[2, 2], &[4.0, 5.0, 10.0, 11.0]);
    assert_close(&transposed, &[2, 2], &[13.0, 20.0, 25.0, 32.0]);
    assert_close(&columns, &[2, 2], &[104.0, 105.0, 210.0, 211.0]);
}

#[test]
fn integer_gemm_truncates_the_scaled_result() {
    let product = op(
        "gemm<a=as_is, b=as_is, alpha=0.5, beta=1.0>",
        "i64[1, 1]",
        &[i64s(&[1, 2], &[1, 2]), i64s(&[2, 1], &[3, -5]), i64s(&[1], &[1])],
    );

    assert_exact(&product, &[1, 1], &[-2]);
}

#[test]
fn every_matrix_of_a_batch_multiplies() {
    let left = f32s(&[2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let right = f32s(&[2, 2], &[1.0, 1.0, 0.0, 1.0]);
    let swaps = f32s(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    let square = f32s(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);

    assert_close(
        &op("matmul", "f32[2, 2, 2]", &[left, right]),
        &[2, 2, 2],
        &[1.0, 3.0, 3.0, 7.0, 5.0, 11.0, 7.0, 15.0],
    );
    assert_close(
        &op("matmul", "f32[2, 2, 2]", &[square, swaps]),
        &[2, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 4.0, 3.0],
    );
}

#[test]
fn an_empty_batch_broadcast_against_one_leaves_the_product_empty() {
    let empty_batch = f32s(&[0, 2, 3], &[]);
    let one_batch = f32s(&[1, 3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let product = op("matmul", "f32[0, 2, 2]", &[empty_batch, one_batch]);

    assert_close(&product, &[0, 2, 2], &[]);
}
