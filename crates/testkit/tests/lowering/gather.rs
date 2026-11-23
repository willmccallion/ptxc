//! `Gather`, `GatherElements` and `GatherND` with valid indices of both
//! index types, negative ones included, on every element type.

use crate::harness::check_against_reference;
use proptest::prelude::*;
use ptxc_core::HostTensor;
use ptxc_graph::Op;
use ptxc_testkit::families::cases::single_node;
use ptxc_testkit::families::gather::{gather_cases, gather_elements_cases, gather_nd_cases};

fn check((op, inputs): (Op, Vec<HostTensor>)) -> Result<(), TestCaseError> {
    let graph = single_node(op, &inputs);
    check_against_reference(graph, &inputs).map_err(TestCaseError::fail)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn gather_agrees_with_the_reference(case in gather_cases()) { check(case)?; }

    #[test]
    fn gather_elements_agrees_with_the_reference(case in gather_elements_cases()) { check(case)?; }

    #[test]
    fn gather_nd_agrees_with_the_reference(case in gather_nd_cases()) { check(case)?; }
}
