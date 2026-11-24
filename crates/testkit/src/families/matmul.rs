//! `MatMul` and `MatMulInteger` with broadcast batches and vector operands,
//! and `Gemm` in every orientation with and without a bias, at every type
//! they take.

use super::values::{Domain, operand_dims, tensor};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::select;
use ptxc_core::{Axis, DType, HostTensor, ZeroPoint};
use ptxc_graph::{F32Attr, GemmAttrs, MatMulIntegerAttrs, Op, Orientation, ZeroPoints};

const TYPES: [DType; 5] = [DType::F32, DType::F16, DType::Bf16, DType::I32, DType::I64];

fn extent() -> impl Strategy<Value = u64> {
    prop_oneof![1 => Just(0_u64), 6 => 1_u64..=3]
}

/// An operand's dims: a vector `[k]`, or its batch then `[rows, columns]`.
fn matrix_dims(batch: &[u64], is_vector: bool, vector: u64, matrix: [u64; 2]) -> Vec<u64> {
    if is_vector { vec![vector] } else { [batch, &matrix[..]].concat() }
}

/// `MatMul` of every rank pairing, batch axes broadcasting.
pub fn matmul_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let shapes = (vec(extent(), 0..=2), [extent(), extent(), extent()], any::<[bool; 2]>());
    (select(TYPES.to_vec()), shapes)
        .prop_flat_map(|(dtype, (batch, [m, k, n], vectors))| {
            let lhs = operand_dims(batch.clone())
                .prop_map(move |b| matrix_dims(&b, vectors[0], k, [m, k]));
            let rhs = operand_dims(batch).prop_map(move |b| matrix_dims(&b, vectors[1], k, [k, n]));
            (lhs, rhs).prop_flat_map(move |(lhs, rhs)| {
                (tensor(dtype, lhs, Domain::Moderate), tensor(dtype, rhs, Domain::Moderate))
            })
        })
        .prop_map(|(lhs, rhs)| (Op::MatMul, vec![lhs, rhs]))
}

const INT8: [DType; 2] = [DType::I8, DType::U8];

/// `MatMulInteger` of every rank pairing, either 8-bit type on each side,
/// zero points absent, per tensor, or per row of `a` and column of `b`.
pub fn matmul_integer_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let shapes = (vec(extent(), 0..=2), [extent(), extent(), extent()], any::<[bool; 2]>());
    let types = (select(INT8.to_vec()), select(INT8.to_vec()));
    (types, shapes)
        .prop_flat_map(|((a_type, b_type), (batch, [m, k, n], vectors))| {
            let lhs = operand_dims(batch.clone())
                .prop_map(move |b| matrix_dims(&b, vectors[0], k, [m, k]));
            let rhs = operand_dims(batch).prop_map(move |b| matrix_dims(&b, vectors[1], k, [k, n]));
            (lhs, rhs).prop_flat_map(move |(lhs, rhs)| {
                let rows = lhs.len().checked_sub(2);
                let columns = (rhs.len() >= 2).then(|| rhs.len() - 1);
                (
                    tensor(a_type, lhs.clone(), Domain::Any),
                    tensor(b_type, rhs.clone(), Domain::Any),
                    zero_points(a_type, &lhs, rows),
                    zero_points(b_type, &rhs, columns),
                )
            })
        })
        .prop_map(|(lhs, rhs, a, b)| {
            (Op::MatMulInteger(MatMulIntegerAttrs { a, b }), vec![lhs, rhs])
        })
}

/// Zero points of `dtype` for an operand of `dims`, often 0: one, or one
/// per index along `axis` where the operand has it.
fn zero_points(dtype: DType, dims: &[u64], axis: Option<usize>) -> BoxedStrategy<ZeroPoints> {
    let range = if dtype == DType::U8 { 0..=255 } else { -128..=127 };
    let point = prop_oneof![Just(0), range].prop_map(ZeroPoint::new);
    let per_tensor = point.clone().prop_map(ZeroPoints::PerTensor);
    let Some(axis) = axis else { return per_tensor.boxed() };
    let count = usize::try_from(dims[axis]).unwrap();
    let per_axis = vec(point, count)
        .prop_map(move |points| ZeroPoints::PerAxis { axis: Axis::new(axis), points });
    prop_oneof![per_tensor, per_axis].boxed()
}

fn orientation() -> impl Strategy<Value = Orientation> {
    prop_oneof![Just(Orientation::AsIs), Just(Orientation::Transposed)]
}

fn factor() -> impl Strategy<Value = F32Attr> {
    select(vec![1.0_f32, 0.5, -2.0, 0.0]).prop_map(F32Attr::new)
}

fn stored(orientation: Orientation, [rows, columns]: [u64; 2]) -> Vec<u64> {
    match orientation {
        Orientation::AsIs => vec![rows, columns],
        Orientation::Transposed => vec![columns, rows],
    }
}

/// Integer `Gemm` scales in f32 (see the NOTE in lowering), so its integers
/// stay small enough to be exact.
pub fn gemm_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let attrs = (orientation(), orientation(), factor(), factor())
        .prop_map(|(a, b, alpha, beta)| GemmAttrs { a, b, alpha, beta });
    (select(TYPES.to_vec()), attrs, [extent(), extent(), extent()], any::<bool>())
        .prop_flat_map(|(dtype, attrs, [m, k, n], has_bias)| {
            let domain = if dtype.is_float() { Domain::Moderate } else { Domain::SmallInt };
            let lhs = tensor(dtype, stored(attrs.a, [m, k]), domain);
            let rhs = tensor(dtype, stored(attrs.b, [k, n]), domain);
            let bias =
                operand_dims(vec![m, n]).prop_flat_map(move |dims| tensor(dtype, dims, domain));
            (Just(attrs), lhs, rhs, vec(bias, usize::from(has_bias)))
        })
        .prop_map(|(attrs, lhs, rhs, bias)| (Op::Gemm(attrs), [vec![lhs, rhs], bias].concat()))
}
