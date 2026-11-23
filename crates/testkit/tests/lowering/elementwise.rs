//! Every elementwise op, `Where` and `Cast`, at every element type its type
//! rule admits, with broadcasting.

use crate::harness::check_against_reference;
use proptest::prelude::*;
use ptxc_testkit::families::cases::graphs;
use ptxc_testkit::families::elementwise::{
    binary_cases, cast_cases, unary_cases, variadic_cases, where_cases,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn unary_ops_agree_with_the_reference((graph, inputs) in graphs(unary_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn binary_ops_agree_with_the_reference((graph, inputs) in graphs(binary_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn variadic_ops_agree_with_the_reference((graph, inputs) in graphs(variadic_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn where_agrees_with_the_reference((graph, inputs) in graphs(where_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn casts_agree_with_the_reference((graph, inputs) in graphs(cast_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }
}
