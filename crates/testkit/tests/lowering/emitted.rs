//! Every op family's kernels emitted as PTX and assembled by ptxas: a few
//! cases drawn from each family's strategy, every stage under its default
//! schedule. And index arithmetic a kernel's loads share, computed once.

use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use ptxc_codegen::generate;
use ptxc_core::float::Bf16;
use ptxc_core::{Dim, HostTensor, Shape, TensorData};
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

/// The output column of an average pool over a 3-wide row is a remainder
/// by 3 that only its loads read, in its window loops and after them to
/// count its taps: one, before the loops.
#[test]
fn an_average_pool_finds_its_output_column_once_before_its_window_loops() {
    let text = "\
graph(%x0: bf16[2, 1, 4, 3]) {
  %y0 = average_pool<pads=include, rounding=ceil, axes=[[2, 2, 1, 1, 0], [2, 1, 2, 2, 1]]> %x0 : bf16[2, 1, 3, 4]
  return %y0
}
";
    let graph = ptxc_graph::text::parse(text, &[]).unwrap();
    let data: TensorData = vec![Bf16::from_f64(1.0); 24].into();
    let input = HostTensor::new(Shape::new([2, 1, 4, 3].map(Dim::new).to_vec()).unwrap(), data);
    let target = preset(SmVersion::SM_89).unwrap();
    let mut modules = Vec::new();

    let _outputs = run_lowered(graph, &[input.unwrap()], |kernel, arguments| {
        modules.push(generate(kernel, &target).unwrap().to_string());
        ptxc_loop::run(kernel, arguments)
    })
    .unwrap();

    let [module] = &modules[..] else { panic!("{} kernels", modules.len()) };
    let lines: Vec<&str> = module.lines().collect();
    let remainders: Vec<usize> = (0..lines.len())
        .filter(|&at| lines[at].contains("rem.u32") && lines[at].ends_with(", 3;"))
        .collect();
    let first_loop = lines.iter().position(|line| line.starts_with("$L__")).unwrap();
    assert_eq!(remainders.len(), 1, "{module}");
    assert!(remainders[0] < first_loop, "{module}");
}
