//! The graphed executor: a program's steps recorded once into a CUDA graph
//! and replayed, giving exactly what the eager executor gives on every
//! run's own inputs.

use crate::device::device;
use crate::programs::{compiled, mlp};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use ptxc_core::{HostTensor, TensorData};
use ptxc_runtime::{Executor, Graphed, RuntimeError, load};
use ptxc_testkit::compare::{Tolerance, compare};
use ptxc_testkit::families;
use ptxc_testkit::sums::interpreted;

const CASES_PER_FAMILY: usize = 4;

/// `program` loaded twice, run eagerly and captured into a graph.
fn both(program: &ptxc_program::Program) -> (Executor<ptxc_runtime::Eager>, Executor<Graphed>) {
    let eager = Executor::eager(load(program.clone(), device()).unwrap());
    let graphed = Executor::eager(load(program.clone(), device()).unwrap()).capture().unwrap();
    (eager, graphed)
}

fn assert_same(graphed: &[HostTensor], eager: &[HostTensor], context: &str) {
    assert_eq!(graphed.len(), eager.len(), "{context}");
    for (graphed, eager) in graphed.iter().zip(eager) {
        compare(graphed, eager, Tolerance::Exact)
            .unwrap_or_else(|mismatch| panic!("{mismatch}\n{context}"));
    }
}

#[test]
fn every_family_runs_graphed_exactly_as_eagerly() {
    let mut runner = TestRunner::deterministic();
    for family in families::all() {
        for _ in 0..CASES_PER_FAMILY {
            let (graph, inputs) = family.graphs.new_tree(&mut runner).unwrap().current();
            if interpreted(graph.clone(), &inputs).is_err() {
                continue;
            }
            let context = format!("family {}\n{}", family.name, ptxc_graph::text::print(&graph));
            let (eager, mut graphed) = both(&compiled(graph));

            let replayed = graphed.run(&inputs).unwrap();

            assert_same(&replayed, &eager.run(&inputs).unwrap(), &context);
        }
    }
}

#[test]
fn a_graph_replays_on_each_runs_own_inputs() {
    let (graph, inputs) = mlp();
    let (eager, mut graphed) = both(&compiled(graph));
    let shifted: Vec<HostTensor> = inputs
        .iter()
        .map(|input| {
            let TensorData::F32(values) = input.data() else { panic!("the MLP takes f32") };
            let shifted: Vec<f32> = values.iter().map(|v| v.mul_add(-1.5, 0.25)).collect();
            HostTensor::new(input.shape().clone(), shifted.into()).unwrap()
        })
        .collect();

    for run in [&inputs, &shifted, &inputs] {
        let replayed = graphed.run(run).unwrap();

        assert_same(&replayed, &eager.run(run).unwrap(), "the MLP");
    }
}

#[test]
fn a_graph_refuses_inputs_of_the_wrong_count_type_or_shape() {
    let x = || HostTensor::vector(vec![1.0_f32, 2.0].into());
    let graph =
        families::cases::single_node(ptxc_graph::Op::Unary(ptxc_graph::UnaryOp::Neg), &[x()]);
    let (_, mut graphed) = both(&compiled(graph));

    let count = graphed.run(&[x(), x()]).unwrap_err();
    let dtype = graphed.run(&[HostTensor::vector(vec![1_i32, 2].into())]).unwrap_err();
    let shape = graphed.run(&[HostTensor::vector(vec![1.0_f32, 2.0, 3.0].into())]).unwrap_err();

    assert!(matches!(count, RuntimeError::InputCount { expected: 1, actual: 2 }), "{count}");
    assert!(matches!(dtype, RuntimeError::Input { .. }), "{dtype}");
    assert!(matches!(shape, RuntimeError::Input { .. }), "{shape}");
    assert_eq!(graphed.run(&[x()]).unwrap().len(), 1);
}
