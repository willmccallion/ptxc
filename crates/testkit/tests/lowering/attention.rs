//! `Attention` in both layouts with grouped key and value heads, every mask
//! kind (a short last axis included), causality, a past, non-pad lengths,
//! softcapping and every QK output.

use crate::harness::check_within;
use proptest::prelude::*;
use ptxc_core::ElementCount;
use ptxc_testkit::compare::Tolerance;
use ptxc_testkit::families::attention::attention_cases;
use ptxc_testkit::families::cases::single_node;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn attention_agrees_with_the_reference((op, inputs) in attention_cases()) {
        let length: u64 = inputs.iter().map(|input| input.shape().element_count().get()).sum();
        let graph = single_node(op, &inputs);
        let length = ElementCount::new(length.max(1));
        let tolerance = |dtype| Tolerance::for_dtype(dtype).for_reduction(length);
        check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)?;
    }
}
