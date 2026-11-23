//! Every activation, `Clip` and `PRelu`, at every element type its type rule
//! admits.

use crate::harness::check_against_reference;
use proptest::prelude::*;
use ptxc_testkit::families::activation::{activation_cases, clip_cases, prelu_cases};
use ptxc_testkit::families::cases::graphs;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn activations_agree_with_the_reference((graph, inputs) in graphs(activation_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn clip_agrees_with_the_reference((graph, inputs) in graphs(clip_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn prelu_agrees_with_the_reference((graph, inputs) in graphs(prelu_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }
}
