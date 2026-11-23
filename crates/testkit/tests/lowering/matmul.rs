//! `MatMul` with broadcast batches and vector operands, and `Gemm` in every
//! orientation with and without a bias, at every type they take.

use crate::cases::single_node;
use crate::harness::check_within;
use crate::values::{Domain, operand_dims, tensor};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::select;
use ptxc_core::{DType, ElementCount, HostTensor};
use ptxc_graph::{F32Attr, GemmAttrs, Op, Orientation};
use ptxc_testkit::compare::Tolerance;

const TYPES: [DType; 5] = [DType::F32, DType::F16, DType::Bf16, DType::I32, DType::I64];

fn extent() -> impl Strategy<Value = u64> {
    prop_oneof![1 => Just(0_u64), 6 => 1_u64..=3]
}

/// An operand's dims: a vector `[k]`, or its batch then `[rows, columns]`.
fn matrix_dims(batch: &[u64], is_vector: bool, vector: u64, matrix: [u64; 2]) -> Vec<u64> {
    if is_vector { vec![vector] } else { [batch, &matrix[..]].concat() }
}

fn matmul_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
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
fn gemm_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
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

fn check((op, inputs): (Op, Vec<HostTensor>)) -> Result<(), TestCaseError> {
    let length = inputs[0].shape().element_count().get().max(1);
    let graph = single_node(op, &inputs);
    let tolerance = |dtype| Tolerance::for_dtype(dtype).for_reduction(ElementCount::new(length));
    check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn matmul_agrees_with_the_reference(case in matmul_cases()) { check(case)?; }

    #[test]
    fn gemm_agrees_with_the_reference(case in gemm_cases()) { check(case)?; }
}
