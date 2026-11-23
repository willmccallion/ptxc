//! `MatMul` and `Gemm`.

use crate::builders::{FLOAT, assert_lines, fixed, float, int, line, node};

#[test]
fn matmul_and_gemm_with_and_without_a_bias() {
    let pair = || vec![fixed("a", FLOAT, &[2, 3]), fixed("b", FLOAT, &[3, 4])];
    let matmul = line(node("MatMul", &["a", "b"], &["y"], vec![]), pair(), vec![]);
    let plain = line(node("Gemm", &["a", "b"], &["y"], vec![]), pair(), vec![]);
    let flipped_b = vec![fixed("a", FLOAT, &[2, 3]), fixed("b", FLOAT, &[4, 3])];
    let attrs = vec![int("transB", 1), float("alpha", 0.5)];
    let flipped = line(node("Gemm", &["a", "b"], &["y"], attrs), flipped_b, vec![]);
    let mut biased_inputs = vec![fixed("a", FLOAT, &[3, 2]), fixed("b", FLOAT, &[3, 4])];
    biased_inputs.push(fixed("c", FLOAT, &[4]));
    let attrs = vec![int("transA", 1), float("beta", 2.0)];
    let biased = line(node("Gemm", &["a", "b", "c"], &["y"], attrs), biased_inputs, vec![]);

    assert_lines(&[
        (matmul, "matmul %a, %b : f32[2, 4]"),
        (plain, "gemm<a=as_is, b=as_is, alpha=1.0, beta=1.0> %a, %b : f32[2, 4]"),
        (flipped, "gemm<a=as_is, b=transposed, alpha=0.5, beta=1.0> %a, %b : f32[2, 4]"),
        (biased, "gemm<a=transposed, b=as_is, alpha=1.0, beta=2.0> %a, %b, %c : f32[2, 4]"),
    ]);
}
