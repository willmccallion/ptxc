//! Convolution and pooling over random windows: strides, dilations,
//! padding, ceil rounding and channel groups.

use crate::harness::check_within;
use proptest::prelude::*;
use ptxc_core::ElementCount;
use ptxc_core::HostTensor;
use ptxc_graph::Op;
use ptxc_testkit::compare::Tolerance;
use ptxc_testkit::families::cases::single_node;
use ptxc_testkit::families::window::{conv_cases, global_pool_cases, pool_cases};

fn check((op, inputs): (Op, Vec<HostTensor>)) -> Result<(), TestCaseError> {
    let length: u64 = inputs.iter().map(|input| input.shape().element_count().get()).sum();
    let graph = single_node(op, &inputs);
    let length = ElementCount::new(length.max(1));
    let tolerance = |dtype| Tolerance::for_dtype(dtype).for_reduction(length);
    check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn pooling_agrees_with_the_reference(case in pool_cases()) { check(case)?; }

    #[test]
    fn global_pooling_agrees_with_the_reference(case in global_pool_cases()) { check(case)?; }

    #[test]
    fn convolution_agrees_with_the_reference(case in conv_cases()) { check(case)?; }
}
