//! Every reduction, arg-reduction and running sum at every element type its
//! type rule admits, over random axes.

use crate::harness::check_within;
use proptest::prelude::*;
use ptxc_testkit::compare::Tolerance;
use ptxc_testkit::families::reduce::{
    arg_reduce_graphs, cum_sum_graphs, reduce_graphs, reduction_tolerance,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn reductions_agree_with_the_reference((graph, inputs) in reduce_graphs()) {
        let tolerance = reduction_tolerance(&inputs[0]);
        check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn arg_reductions_agree_with_the_reference((graph, inputs) in arg_reduce_graphs()) {
        check_within(graph, &inputs, Tolerance::for_dtype).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn running_sums_agree_with_the_reference((graph, inputs) in cum_sum_graphs()) {
        let tolerance = reduction_tolerance(&inputs[0]);
        check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)?;
    }
}
