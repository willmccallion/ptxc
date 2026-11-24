//! `MatMul` and `MatMulInteger` with broadcast batches and vector operands,
//! and `Gemm` in every orientation with and without a bias, at every type
//! they take.

use crate::harness::check_within;
use proptest::prelude::*;
use ptxc_core::{Dim, ElementCount, Evaluate, HostTensor, Shape, TensorData, ZeroPoint};
use ptxc_graph::{MatMulIntegerAttrs, Op, ZeroPoints};
use ptxc_interp::Interpreter;
use ptxc_testkit::compare::Tolerance;
use ptxc_testkit::families::cases::single_node;
use ptxc_testkit::families::matmul::{gemm_cases, matmul_cases, matmul_integer_cases};
use ptxc_testkit::sums::{SumRounding, interpreted};

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

    #[test]
    fn matmul_integer_agrees_with_the_reference(case in matmul_integer_cases()) { check(case)?; }
}

fn u8_matrix(dims: [u64; 2], values: Vec<u8>) -> HostTensor {
    HostTensor::new(Shape::new(dims.map(Dim::new).to_vec()).unwrap(), values.into()).unwrap()
}

/// ONNX's `test_matmulinteger`, whose opset 10 the importer does not take,
/// though `MatMulInteger` is unchanged since.
#[test]
fn onnx_s_matmul_integer_example_lowers_onto_tensor_cores_and_matches() {
    let inputs = [
        u8_matrix([4, 3], vec![11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0]),
        u8_matrix([3, 2], vec![1, 4, 2, 5, 3, 6]),
    ];
    let point = |value| ZeroPoints::PerTensor(ZeroPoint::new(value));
    let op = Op::MatMulInteger(MatMulIntegerAttrs { a: point(12), b: point(0) });
    let graph = single_node(op, &inputs);
    let expected = [-38, -83, -44, -98, -50, -113, -56, -128];

    let reference = Interpreter::new(&graph).evaluate(&inputs).unwrap();
    let (lowered, rounding) = interpreted(graph, &inputs).unwrap();

    for outputs in [reference, lowered] {
        let TensorData::I32(values) = outputs[0].data() else { panic!("{outputs:?}") };
        assert_eq!(values, &expected);
    }
    assert!(matches!(rounding, SumRounding::TensorCores { .. }));
}
