use crate::common::{f32_type, shape, tensor_type};
use ptxc_core::{Axis, DType, Dim, Rank, ZeroPoint};
use ptxc_graph::{
    F32Attr, GemmAttrs, MatMulIntegerAttrs, Op, Orientation, TensorType, TypeError, TypeRule,
    ZeroPoints,
};

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

const fn per_tensor(point: i32) -> ZeroPoints {
    ZeroPoints::PerTensor(ZeroPoint::new(point))
}

fn per_axis(axis: usize, points: &[i32]) -> ZeroPoints {
    let points = points.iter().copied().map(ZeroPoint::new).collect();
    ZeroPoints::PerAxis { axis: Axis::new(axis), points }
}

fn matmul_integer(
    (a, b): (ZeroPoints, ZeroPoints),
    [lhs, rhs]: &[TensorType; 2],
) -> Result<Vec<TensorType>, TypeError> {
    Op::MatMulInteger(MatMulIntegerAttrs { a, b }).infer(&[lhs, rhs])
}

#[test]
fn matmul_integer_gives_i32_products_of_either_8_bit_type() {
    let points = || (per_axis(1, &[0, 1, 2]), per_axis(1, &[255, 0]));
    let batched = [tensor_type(DType::I8, &[2, 3, 4]), tensor_type(DType::U8, &[4, 2])];
    let vectors = [tensor_type(DType::U8, &[4]), tensor_type(DType::I8, &[4])];

    let product = matmul_integer(points(), &batched);
    let dot = matmul_integer((per_tensor(3), per_tensor(-5)), &vectors);

    assert_eq!(product, Ok(vec![tensor_type(DType::I32, &[2, 3, 2])]));
    assert_eq!(dot, Ok(vec![tensor_type(DType::I32, &[])]));
}

#[test]
fn matmul_integer_zero_points_vary_by_rows_of_a_and_columns_of_b_in_range() {
    let operands = || [tensor_type(DType::I8, &[3, 4]), tensor_type(DType::U8, &[4, 2])];
    let along_depth = matmul_integer((per_axis(1, &[0; 4]), per_tensor(0)), &operands());
    let too_few = matmul_integer((per_tensor(0), per_axis(1, &[1])), &operands());
    let negative_u8 = matmul_integer((per_tensor(0), per_tensor(-1)), &operands());
    let too_big_i8 = matmul_integer((per_tensor(128), per_tensor(0)), &operands());
    let floats =
        matmul_integer((per_tensor(0), per_tensor(0)), &[f32_type(&[3, 4]), f32_type(&[4, 2])]);

    assert_eq!(along_depth, Err(TypeError::QuantAxis(Axis::new(1))));
    let (axis, dim) = (Axis::new(1), Dim::new(2));
    assert_eq!(too_few, Err(TypeError::QuantMapCount { axis, dim, maps: 1 }));
    let range =
        |point, dtype| Err(TypeError::ZeroPointRange { zero_point: ZeroPoint::new(point), dtype });
    assert_eq!(negative_u8, range(-1, DType::U8));
    assert_eq!(too_big_i8, range(128, DType::I8));
    assert_eq!(floats, Err(TypeError::UnsupportedDType(DType::F32)));
}
