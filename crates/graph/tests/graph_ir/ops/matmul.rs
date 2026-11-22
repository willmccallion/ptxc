use crate::common::{f32_type, shape, tensor_type};
use ptxc_core::{DType, Dim, Rank};
use ptxc_graph::{F32Attr, GemmAttrs, Op, Orientation, TensorType, TypeError, TypeRule};

fn matmul(a: &[u64], b: &[u64]) -> Result<Vec<TensorType>, TypeError> {
    Op::MatMul.infer(&[&f32_type(a), &f32_type(b)])
}

const fn gemm(a: Orientation, b: Orientation) -> GemmAttrs {
    GemmAttrs { a, b, alpha: F32Attr::new(1.0), beta: F32Attr::new(1.0) }
}

const PLAIN: GemmAttrs = gemm(Orientation::AsIs, Orientation::AsIs);

#[test]
fn matmul_multiplies_matrices() {
    assert_eq!(matmul(&[2, 3], &[3, 4]), Ok(vec![f32_type(&[2, 4])]));
}

#[test]
fn matmul_broadcasts_batch_axes() {
    assert_eq!(matmul(&[5, 1, 2, 3], &[4, 3, 6]), Ok(vec![f32_type(&[5, 4, 2, 6])]));
}

#[test]
fn matmul_promotes_vectors_and_drops_the_added_axis() {
    assert_eq!(matmul(&[3], &[2, 3, 4]), Ok(vec![f32_type(&[2, 4])]));
    assert_eq!(matmul(&[2, 3, 4], &[4]), Ok(vec![f32_type(&[2, 3])]));
    assert_eq!(matmul(&[3], &[3]), Ok(vec![f32_type(&[])]));
}

#[test]
fn matmul_rejects_unequal_contracted_dims() {
    assert_eq!(
        matmul(&[2, 3], &[5, 4]),
        Err(TypeError::ContractedDims { lhs: Dim::new(3), rhs: Dim::new(5) })
    );
    assert!(matmul(&[3], &[4]).is_err());
}

#[test]
fn matmul_rejects_scalars() {
    let error = matmul(&[], &[3, 4]);

    assert_eq!(error, Err(TypeError::RankBelow { minimum: Rank::new(1), actual: Rank::new(0) }));
    assert!(matmul(&[3, 4], &[]).is_err());
}

#[test]
fn matmul_rejects_batch_axes_that_do_not_broadcast() {
    let error = matmul(&[2, 2, 3], &[5, 3, 4]);

    assert_eq!(error, Err(TypeError::NotBroadcastable { lhs: shape(&[2]), rhs: shape(&[5]) }));
}

#[test]
fn matmul_takes_wide_integers_but_not_narrow_ones() {
    let wide = tensor_type(DType::I64, &[2, 2]);
    let narrow = tensor_type(DType::U8, &[2, 2]);

    assert_eq!(Op::MatMul.infer(&[&wide, &wide]), Ok(vec![wide.clone()]));
    assert_eq!(Op::MatMul.infer(&[&narrow, &narrow]), Err(TypeError::UnsupportedDType(DType::U8)));
    assert_eq!(
        Op::MatMul.infer(&[&f32_type(&[2, 2]), &tensor_type(DType::F16, &[2, 2])]),
        Err(TypeError::MixedDTypes { lhs: DType::F32, rhs: DType::F16 })
    );
}

#[test]
fn gemm_multiplies_matrices_in_their_orientations() {
    let expected = Ok(vec![f32_type(&[2, 4])]);

    assert_eq!(PLAIN.infer(&[&f32_type(&[2, 3]), &f32_type(&[3, 4])]), expected);
    let transposed_a = gemm(Orientation::Transposed, Orientation::AsIs);
    assert_eq!(transposed_a.infer(&[&f32_type(&[3, 2]), &f32_type(&[3, 4])]), expected);
    let transposed_b = gemm(Orientation::AsIs, Orientation::Transposed);
    assert_eq!(transposed_b.infer(&[&f32_type(&[2, 3]), &f32_type(&[4, 3])]), expected);
}

#[test]
fn gemm_rejects_unequal_contracted_dims() {
    let error = PLAIN.infer(&[&f32_type(&[2, 3]), &f32_type(&[4, 3])]);

    assert_eq!(error, Err(TypeError::ContractedDims { lhs: Dim::new(3), rhs: Dim::new(4) }));
}

#[test]
fn gemm_bias_broadcasts_to_the_product() {
    let (a, b) = (f32_type(&[2, 3]), f32_type(&[3, 4]));

    for bias in [f32_type(&[4]), f32_type(&[2, 1]), f32_type(&[2, 4]), f32_type(&[])] {
        assert_eq!(PLAIN.infer(&[&a, &b, &bias]), Ok(vec![f32_type(&[2, 4])]), "{bias:?}");
    }
}

#[test]
fn gemm_bias_that_does_not_broadcast_to_the_product_is_rejected() {
    let (a, b) = (f32_type(&[2, 3]), f32_type(&[3, 4]));

    let error = PLAIN.infer(&[&a, &b, &f32_type(&[3])]);

    assert_eq!(error, Err(TypeError::NotUnidirectional { from: shape(&[3]), to: shape(&[2, 4]) }));
    assert_eq!(
        PLAIN.infer(&[&a, &b, &tensor_type(DType::F16, &[4])]),
        Err(TypeError::MixedDTypes { lhs: DType::F32, rhs: DType::F16 })
    );
}

#[test]
fn gemm_takes_only_matrices() {
    let error = PLAIN.infer(&[&f32_type(&[1, 2, 3]), &f32_type(&[3, 4])]);

    assert_eq!(error, Err(TypeError::Rank { expected: Rank::new(2), actual: Rank::new(3) }));
    assert!(PLAIN.infer(&[&f32_type(&[2, 3]), &f32_type(&[3])]).is_err());
}

#[test]
fn gemm_takes_two_or_three_operands() {
    let operand = f32_type(&[2, 2]);

    for operands in [vec![&operand], vec![&operand; 4]] {
        let error = PLAIN.infer(&operands).unwrap_err();
        assert!(
            matches!(error, TypeError::ArityBetween { min, max, actual }
                if min.get() == 2 && max.get() == 3 && actual.get() == operands.len()),
            "{error}"
        );
    }
}
