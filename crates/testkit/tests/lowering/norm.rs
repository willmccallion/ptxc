//! Softmax and every normalisation, at every float type, over random axes
//! and parameter shapes.

use crate::harness::check_within;
use proptest::prelude::*;
use ptxc_testkit::families::norm::{
    batch_norm_graphs, instance_norm_graphs, layer_norm_graphs, softmax_graphs, tolerance_for,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn softmax_agrees_with_the_reference((graph, inputs) in softmax_graphs()) {
        let tolerance = tolerance_for(&inputs[0]);
        check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn layer_norm_agrees_with_the_reference((graph, inputs) in layer_norm_graphs()) {
        let tolerance = tolerance_for(&inputs[0]);
        check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn batch_norm_agrees_with_the_reference((graph, inputs) in batch_norm_graphs()) {
        let tolerance = tolerance_for(&inputs[0]);
        check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn instance_norm_agrees_with_the_reference((graph, inputs) in instance_norm_graphs()) {
        let tolerance = tolerance_for(&inputs[0]);
        check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)?;
    }
}
