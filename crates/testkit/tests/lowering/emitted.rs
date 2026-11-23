//! Every op family's kernels emitted as PTX and assembled by ptxas: a few
//! cases drawn from each family's strategy, every stage under its default
//! schedule.

use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use ptxc_codegen::generate;
use ptxc_core::HostTensor;
use ptxc_graph::Graph;
use ptxc_target::{SmVersion, preset};
use ptxc_testkit::families;
use ptxc_testkit::lowered::run_lowered;

/// Graphs drawn per family; each stage is one ptxas run.
const CASES_PER_FAMILY: usize = 6;

/// Emits and assembles every stage of `graph`, each run in the loop
/// interpreter for the next stage's inputs.
fn assemble_stages(family: &str, (graph, inputs): (Graph, Vec<HostTensor>)) {
    let target = preset(SmVersion::SM_89).unwrap();
    let gpu = target.sm().to_string();
    let ran = run_lowered(graph, &inputs, |kernel, arguments| {
        let module = generate(kernel, &target).unwrap_or_else(|error| panic!("{family}: {error}"));
        if let Err(error) = ptxc_testkit::ptxas::assemble(&module.to_string(), &gpu) {
            panic!("{family}: {error}");
        }
        ptxc_loop::run(kernel, arguments)
    });
    if let Err(error) = ran {
        panic!("{family}: {error}");
    }
}

#[test]
fn every_family_emits_kernels_ptxas_accepts() {
    let mut runner = TestRunner::deterministic();
    for family in families::all() {
        for _ in 0..CASES_PER_FAMILY {
            let case = family.graphs.new_tree(&mut runner).unwrap().current();
            assemble_stages(family.name, case);
        }
    }
}
