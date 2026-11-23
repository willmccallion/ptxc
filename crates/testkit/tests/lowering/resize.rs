//! `Resize` in every mode, coordinate transform and antialias setting, on
//! finite inputs: the reference multiplies zero-weight taps, so a NaN or an
//! infinity beside a selected element spreads there, which ptxc's direct
//! reads do not (D47).

use crate::harness::check_within;
use proptest::prelude::*;
use ptxc_core::DType;
use ptxc_core::ElementCount;
use ptxc_testkit::compare::Tolerance;
use ptxc_testkit::families::cases::single_node;
use ptxc_testkit::families::resize::resize_cases;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn resize_agrees_with_the_reference((op, inputs) in resize_cases()) {
        let length = ElementCount::new(inputs[0].shape().element_count().get().max(1));
        let graph = single_node(op, &inputs);
        let tolerance = |dtype: DType| {
            if dtype.is_float() { Tolerance::for_dtype(dtype).for_reduction(length) } else { Tolerance::QUANTIZED }
        };
        check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)?;
    }
}
