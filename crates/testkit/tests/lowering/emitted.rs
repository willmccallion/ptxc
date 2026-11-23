//! Every op family's kernels emitted as PTX and assembled by ptxas: a few
//! cases drawn from each family's strategy, every stage under its default
//! schedule.

use crate::cases::{graphs, single_node};
use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use ptxc_codegen::{CodegenError, generate};
use ptxc_core::HostTensor;
use ptxc_graph::{Graph, Op, Partitioned};
use ptxc_target::{SmVersion, preset};
use std::sync::Arc;

/// Graphs drawn per family; each stage is one ptxas run.
const CASES_PER_FAMILY: usize = 6;

/// What codegen does not emit yet: the remaining math programs.
const PENDING: [&str; 8] = ["expm1", "log1p", "erf", "sin", "cos", "tanh", "pow", "truncated_rem"];

fn from_cases(
    cases: impl Strategy<Value = (Op, Vec<HostTensor>)> + 'static,
) -> BoxedStrategy<Graph> {
    cases.prop_map(|(op, inputs)| single_node(op, &inputs)).boxed()
}

fn from_graphs(
    cases: impl Strategy<Value = (Graph, Vec<HostTensor>)> + 'static,
) -> BoxedStrategy<Graph> {
    cases.prop_map(|(graph, _)| graph).boxed()
}

fn families() -> Vec<(&'static str, BoxedStrategy<Graph>)> {
    use crate::{activation as a, attention as att, elementwise as e, gather as g, matmul as m};
    use crate::{norm as n, quant as q, reduce as r, resize as rs, shape as s, window as w};
    vec![
        ("unary", from_graphs(graphs(e::unary_cases()))),
        ("binary", from_graphs(graphs(e::binary_cases()))),
        ("variadic", from_graphs(graphs(e::variadic_cases()))),
        ("where", from_graphs(graphs(e::where_cases()))),
        ("cast", from_graphs(graphs(e::cast_cases()))),
        ("activation", from_graphs(graphs(a::activation_cases()))),
        ("clip", from_graphs(graphs(a::clip_cases()))),
        ("prelu", from_graphs(graphs(a::prelu_cases()))),
        ("reduce", from_graphs(r::reduce_graphs())),
        ("arg_reduce", from_graphs(r::arg_reduce_graphs())),
        ("cum_sum", from_graphs(r::cum_sum_graphs())),
        ("softmax", from_graphs(n::softmax_graphs())),
        ("layer_norm", from_graphs(n::layer_norm_graphs())),
        ("batch_norm", from_graphs(n::batch_norm_graphs())),
        ("instance_norm", from_graphs(n::instance_norm_graphs())),
        ("reshape", from_cases(s::reshape_cases())),
        ("expand", from_cases(s::expand_cases())),
        ("transpose", from_cases(s::transpose_cases())),
        ("tile", from_cases(s::tile_cases())),
        ("concat", from_cases(s::concat_cases())),
        ("slice", from_cases(s::slice_cases())),
        ("split", from_cases(s::split_cases())),
        ("pad", from_cases(s::pad_cases())),
        ("trilu", from_cases(s::trilu_cases())),
        ("gather", from_cases(g::gather_cases())),
        ("gather_elements", from_cases(g::gather_elements_cases())),
        ("gather_nd", from_cases(g::gather_nd_cases())),
        ("matmul", from_cases(m::matmul_cases())),
        ("gemm", from_cases(m::gemm_cases())),
        ("pool", from_cases(w::pool_cases())),
        ("global_pool", from_cases(w::global_pool_cases())),
        ("conv", from_cases(w::conv_cases())),
        ("quantize", from_cases(q::quantize_cases())),
        ("dequantize", from_cases(q::dequantize_cases())),
        ("qmatmul", from_cases(q::qmatmul_cases())),
        ("qconv", from_cases(q::qconv_cases())),
        ("resize", from_cases(rs::resize_cases())),
        ("attention", from_cases(att::attention_cases())),
    ]
}

/// Emits and assembles every stage of `graph`; codegen may only decline
/// the pending operations.
fn assemble_stages(family: &str, graph: Graph) {
    let target = preset(SmVersion::SM_89).unwrap();
    let gpu = target.sm().to_string();
    let partitioned = Partitioned::one_node_per_group(graph).unwrap();
    for group in ptxc_lower::lower(&partitioned).unwrap() {
        for stage in group.stages() {
            let schedule = ptxc_schedule::default_schedule(Arc::clone(stage.compute()), &target);
            let kernel = schedule.unwrap().lower();
            match generate(&kernel, &target) {
                Ok(module) => {
                    if let Err(error) = ptxc_testkit::ptxas::assemble(&module.to_string(), &gpu) {
                        panic!("{family}: {error}");
                    }
                }
                Err(CodegenError::Unsupported { op }) if PENDING.contains(&op) => {}
                Err(error) => panic!("{family}: {error}"),
            }
        }
    }
}

#[test]
fn every_family_emits_kernels_ptxas_accepts() {
    let mut runner = TestRunner::deterministic();
    for (family, strategy) in families() {
        for _ in 0..CASES_PER_FAMILY {
            let graph = strategy.new_tree(&mut runner).unwrap().current();
            assemble_stages(family, graph);
        }
    }
}
