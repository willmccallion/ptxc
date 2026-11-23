//! `MatMul` with broadcast batches and vector operands, and `Gemm` in every
//! orientation with and without a bias, at every type they take.

use crate::harness::check_within;
use proptest::prelude::*;
use ptxc_core::ElementCount;
use ptxc_core::HostTensor;
use ptxc_graph::Op;
use ptxc_testkit::compare::Tolerance;
use ptxc_testkit::families::cases::single_node;
use ptxc_testkit::families::matmul::{gemm_cases, matmul_cases};

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
