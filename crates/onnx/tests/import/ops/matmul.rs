//! `MatMul`, `MatMulInteger` and `Gemm`.

use crate::builders::{FLOAT, assert_lines, fixed, float, import, int, line, model, node, typed};
use ptxc_onnx::Unsupported;
use ptxc_onnx_proto::tensor_proto::DataType;

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

fn u8_tensor(name: &str, dims: &[i64], values: &[i32]) -> ptxc_onnx_proto::TensorProto {
    typed(name, DataType::Uint8, dims, values)
}

/// `MatMulInteger` on `u8[4, 3]` and `u8[3, 2]` with `zero_points`, by
/// input name.
fn matmul_integer(names: &[&str], zero_points: Vec<ptxc_onnx_proto::TensorProto>) -> String {
    let inputs = vec![fixed("a", DataType::Uint8, &[4, 3]), fixed("b", DataType::Uint8, &[3, 2])];
    line(node("MatMulInteger", names, &["y"], vec![]), inputs, zero_points)
}

#[test]
fn matmul_integer_takes_zero_points_per_tensor_or_per_row_and_column() {
    let onnx_example = matmul_integer(
        &["a", "b", "az", "bz"],
        vec![u8_tensor("az", &[1], &[12]), u8_tensor("bz", &[1], &[0])],
    );
    let per_row_and_column = matmul_integer(
        &["a", "b", "az", "bz"],
        vec![u8_tensor("az", &[4], &[1, 2, 3, 4]), u8_tensor("bz", &[2], &[5, 6])],
    );
    let b_only = matmul_integer(&["a", "b", "", "bz"], vec![u8_tensor("bz", &[], &[7])]);
    let none = matmul_integer(&["a", "b"], vec![]);

    assert_lines(&[
        (onnx_example, "matmul_integer<a=[12], b=[0]> %a, %b : i32[4, 2]"),
        (
            per_row_and_column,
            "matmul_integer<a=[0, [1, 2, 3, 4]], b=[1, [5, 6]]> %a, %b : i32[4, 2]",
        ),
        (b_only, "matmul_integer<a=[0], b=[7]> %a, %b : i32[4, 2]"),
        (none, "matmul_integer<a=[0], b=[0]> %a, %b : i32[4, 2]"),
    ]);
}

#[test]
fn matmul_integer_zero_points_varying_over_a_batch_or_a_vector_are_unsupported() {
    let batched =
        vec![fixed("a", DataType::Uint8, &[2, 4, 3]), fixed("b", DataType::Uint8, &[3, 2])];
    let by_batch = vec![u8_tensor("az", &[2, 4, 1], &[0, 1, 2, 3, 4, 5, 6, 7])];
    let vector = vec![fixed("a", DataType::Uint8, &[4, 3]), fixed("b", DataType::Uint8, &[3])];
    let along_vector = vec![u8_tensor("bz", &[3], &[1, 2, 3])];
    let matmul = |names: &[&str]| vec![node("MatMulInteger", names, &["y"], vec![])];

    let errors = [
        import(&model(matmul(&["a", "b", "az"]), batched, by_batch, vec![])).unwrap_err(),
        import(&model(matmul(&["a", "b", "", "bz"]), vector, along_vector, vec![])).unwrap_err(),
    ];

    for error in errors {
        assert_eq!(
            error.unsupported(),
            Some(&Unsupported::Attribute {
                op: "MatMulInteger".to_owned(),
                attribute: "zero points varying other than by row of a or column of b".to_owned(),
            })
        );
    }
}
