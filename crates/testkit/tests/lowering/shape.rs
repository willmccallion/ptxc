//! Every data-movement op at random shapes and attributes, on every element
//! type.

use crate::harness::check_against_reference;
use proptest::prelude::*;
use ptxc_core::HostTensor;
use ptxc_graph::Op;
use ptxc_testkit::families::shape::{
    concat_cases, expand_cases, graph_of, pad_cases, reshape_cases, slice_cases, split_cases,
    tile_cases, transpose_cases, trilu_cases,
};

fn check(case: (Op, Vec<HostTensor>)) -> Result<(), TestCaseError> {
    let (graph, inputs) = graph_of(case);
    check_against_reference(graph, &inputs).map_err(TestCaseError::fail)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn reshape_agrees_with_the_reference(case in reshape_cases()) { check(case)?; }

    #[test]
    fn expand_agrees_with_the_reference(case in expand_cases()) { check(case)?; }

    #[test]
    fn transpose_agrees_with_the_reference(case in transpose_cases()) { check(case)?; }

    #[test]
    fn tile_agrees_with_the_reference(case in tile_cases()) { check(case)?; }

    #[test]
    fn concat_agrees_with_the_reference(case in concat_cases()) { check(case)?; }

    #[test]
    fn slice_agrees_with_the_reference(case in slice_cases()) { check(case)?; }

    #[test]
    fn split_agrees_with_the_reference(case in split_cases()) { check(case)?; }

    #[test]
    fn pad_agrees_with_the_reference(case in pad_cases()) { check(case)?; }

    #[test]
    fn trilu_agrees_with_the_reference(case in trilu_cases()) { check(case)?; }
}
