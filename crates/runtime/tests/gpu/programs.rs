//! Whole programs: every op family's random cases compiled, saved and
//! reloaded, loaded and run by the eager executor, matching the loop interpreter run stage by stage
//! exactly.

use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use ptxc_compiler::Compilation;
use ptxc_core::{Evaluate, HostTensor};
use ptxc_cuda::{DeviceOrdinal, Driver};
use ptxc_runtime::{Executor, RuntimeError, load};
use ptxc_target::{SmVersion, preset};
use ptxc_testkit::compare::{Tolerance, compare};
use ptxc_testkit::families;
use ptxc_testkit::lowered::run_lowered;

/// Cases drawn per family.
const CASES_PER_FAMILY: usize = 8;

fn run_program(graph: ptxc_graph::Graph, inputs: &[HostTensor]) -> Vec<HostTensor> {
    let target = preset(SmVersion::SM_89).unwrap();
    let compilation = Compilation::from_graph(graph, target);
    let program = compilation.optimize().unwrap().partition().unwrap().lower().unwrap();
    let program = program.schedule().unwrap().emit().unwrap();
    let saved = ptxc_program::load(&ptxc_program::save(&program).unwrap()).unwrap();
    let device = Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap();
    let mut executor = Executor::eager(load(saved, &device).unwrap());
    executor.evaluate(inputs).unwrap()
}

#[test]
fn every_family_compiles_to_a_program_that_matches_the_interpreter() {
    let mut runner = TestRunner::deterministic();
    for family in families::all() {
        let mut compared = 0;
        for _ in 0..CASES_PER_FAMILY {
            let (graph, inputs) = family.graphs.new_tree(&mut runner).unwrap().current();
            let Ok(expected) = run_lowered(graph.clone(), &inputs, ptxc_loop::run) else {
                continue;
            };
            let context = format!("family {}\n{}", family.name, ptxc_graph::text::print(&graph));
            let actual = run_program(graph, &inputs);
            assert_eq!(actual.len(), expected.len(), "{context}");
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                if let Err(mismatch) = compare(actual, expected, Tolerance::Exact) {
                    panic!("output {index} differs from the interpreter:\n{mismatch}\n{context}");
                }
            }
            compared += 1;
        }
        assert!(compared > 0, "{}: the interpreter rejected every case", family.name);
    }
}

/// `program`'s file with its header's target rewritten to `sm_90`.
fn retargeted(program: &ptxc_program::Program) -> Vec<u8> {
    let bytes = ptxc_program::save(program).unwrap();
    let length = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().unwrap())).unwrap();
    let header = String::from_utf8(bytes[16..16 + length].to_vec()).unwrap();
    let header = header.replace("\"target\":[8,9]", "\"target\":[9,0]");
    let mut rewritten = bytes[..16].to_vec();
    rewritten.extend_from_slice(header.as_bytes());
    rewritten.extend_from_slice(&bytes[16 + length..]);
    rewritten
}

#[test]
fn a_program_for_another_gpu_is_refused() {
    let graph = families::cases::single_node(
        ptxc_graph::Op::Unary(ptxc_graph::UnaryOp::Neg),
        &[HostTensor::vector(vec![1.0_f32].into())],
    );
    let target = preset(SmVersion::SM_89).unwrap();
    let program = Compilation::from_graph(graph, target).optimize().unwrap().partition().unwrap();
    let program = program.lower().unwrap().schedule().unwrap().emit().unwrap();
    let foreign = ptxc_program::load(&retargeted(&program)).unwrap();
    let device = Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap();

    let error = load(foreign, &device).unwrap_err();

    assert!(matches!(error, RuntimeError::Target { .. }), "{error}");
}

#[test]
fn inputs_of_the_wrong_count_type_or_shape_are_refused() {
    let x = || HostTensor::vector(vec![1.0_f32, 2.0].into());
    let graph =
        families::cases::single_node(ptxc_graph::Op::Unary(ptxc_graph::UnaryOp::Neg), &[x()]);
    let target = preset(SmVersion::SM_89).unwrap();
    let program = Compilation::from_graph(graph, target).optimize().unwrap().partition().unwrap();
    let program = program.lower().unwrap().schedule().unwrap().emit().unwrap();
    let device = Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap();
    let mut executor = Executor::eager(load(program, &device).unwrap());
    let wrong_type = HostTensor::vector(vec![1_i32, 2].into());
    let wrong_shape = HostTensor::vector(vec![1.0_f32, 2.0, 3.0].into());

    let count = executor.evaluate(&[x(), x()]).unwrap_err();
    let dtype = executor.evaluate(&[wrong_type]).unwrap_err();
    let shape = executor.evaluate(&[wrong_shape]).unwrap_err();

    assert!(matches!(count, RuntimeError::InputCount { expected: 1, actual: 2 }), "{count}");
    assert!(matches!(dtype, RuntimeError::Input { .. }), "{dtype}");
    assert!(matches!(shape, RuntimeError::Input { .. }), "{shape}");
    assert_eq!(executor.evaluate(&[x()]).unwrap().len(), 1);
}
