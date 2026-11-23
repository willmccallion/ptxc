//! Every op family's random cases, lowered and run stage by stage on the
//! GPU, each stage matching the loop interpreter exactly.

use crate::device::match_interpreter;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use ptxc_testkit::families;
use ptxc_testkit::lowered::{LoweredError, run_lowered};

/// Cases drawn per family.
const CASES_PER_FAMILY: usize = 24;

#[test]
fn every_family_matches_the_interpreter_on_the_gpu() {
    let mut runner = TestRunner::deterministic();
    for family in families::all() {
        let mut compared = 0;
        for _ in 0..CASES_PER_FAMILY {
            let (graph, inputs) = family.graphs.new_tree(&mut runner).unwrap().current();
            let context = format!(
                "family {}\n{}\ninputs: {inputs:?}",
                family.name,
                ptxc_graph::text::print(&graph)
            );
            let ran = run_lowered(graph, &inputs, |kernel, arguments| {
                match_interpreter(kernel, arguments, &context)
            });
            match ran {
                Ok(_) => compared += 1,
                Err(LoweredError::Stage(_)) => {}
                Err(error) => panic!("{error}\n{context}"),
            }
        }
        eprintln!("{}: {compared} of {CASES_PER_FAMILY} cases compared", family.name);
        assert!(compared > 0, "{}: the interpreter rejected every case", family.name);
    }
}
