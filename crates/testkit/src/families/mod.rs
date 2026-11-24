//! Generators of single-op graphs and inputs for every op family, reaching
//! each op's edge cases: the lowering tests check them against the
//! reference interpreter, and the GPU tests run their kernels.
//!
//! NOTE: these are test code shared as a library. A generator that builds
//! an invalid shape or graph has a bug, so it panics rather than returning
//! an error a strategy has no way to report (D52).

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    reason = "test generators panic on cases they build wrong (D52)"
)]

pub mod activation;
pub mod attention;
pub mod cases;
pub mod elementwise;
pub mod gather;
pub mod matmul;
pub mod norm;
pub mod quant;
pub mod reduce;
pub mod resize;
pub mod shape;
pub mod values;
pub mod window;

use cases::{graphs, single_node};
use proptest::strategy::{BoxedStrategy, Strategy};
use ptxc_core::HostTensor;
use ptxc_graph::{Graph, Op};

/// A family's name and its graphs with inputs.
#[derive(Debug)]
pub struct Family {
    /// The family, as its tests name it.
    pub name: &'static str,
    /// Single-op graphs of the family, with inputs.
    pub graphs: BoxedStrategy<(Graph, Vec<HostTensor>)>,
}

fn of_cases(
    name: &'static str,
    cases: impl Strategy<Value = (Op, Vec<HostTensor>)> + 'static,
) -> Family {
    let graphs = cases.prop_map(|(op, inputs)| (single_node(op, &inputs), inputs)).boxed();
    Family { name, graphs }
}

fn of_graphs(
    name: &'static str,
    graphs: impl Strategy<Value = (Graph, Vec<HostTensor>)> + 'static,
) -> Family {
    Family { name, graphs: graphs.boxed() }
}

/// Every op family.
pub fn all() -> Vec<Family> {
    use self::{activation as a, attention as att, elementwise as e, gather as g, matmul as m};
    use self::{norm as n, quant as q, reduce as r, resize as rs, shape as s, window as w};
    vec![
        of_graphs("unary", graphs(e::unary_cases())),
        of_graphs("binary", graphs(e::binary_cases())),
        of_graphs("variadic", graphs(e::variadic_cases())),
        of_graphs("where", graphs(e::where_cases())),
        of_graphs("cast", graphs(e::cast_cases())),
        of_graphs("activation", graphs(a::activation_cases())),
        of_graphs("clip", graphs(a::clip_cases())),
        of_graphs("prelu", graphs(a::prelu_cases())),
        of_graphs("reduce", r::reduce_graphs()),
        of_graphs("arg_reduce", r::arg_reduce_graphs()),
        of_graphs("cum_sum", r::cum_sum_graphs()),
        of_graphs("softmax", n::softmax_graphs()),
        of_graphs("layer_norm", n::layer_norm_graphs()),
        of_graphs("batch_norm", n::batch_norm_graphs()),
        of_graphs("instance_norm", n::instance_norm_graphs()),
        of_cases("reshape", s::reshape_cases()),
        of_cases("expand", s::expand_cases()),
        of_cases("transpose", s::transpose_cases()),
        of_cases("tile", s::tile_cases()),
        of_cases("concat", s::concat_cases()),
        of_cases("slice", s::slice_cases()),
        of_cases("split", s::split_cases()),
        of_cases("pad", s::pad_cases()),
        of_cases("trilu", s::trilu_cases()),
        of_cases("gather", g::gather_cases()),
        of_cases("gather_elements", g::gather_elements_cases()),
        of_cases("gather_nd", g::gather_nd_cases()),
        of_cases("matmul", m::matmul_cases()),
        of_cases("matmul_integer", m::matmul_integer_cases()),
        of_cases("gemm", m::gemm_cases()),
        of_cases("pool", w::pool_cases()),
        of_cases("global_pool", w::global_pool_cases()),
        of_cases("conv", w::conv_cases()),
        of_cases("quantize", q::quantize_cases()),
        of_cases("dequantize", q::dequantize_cases()),
        of_cases("qmatmul", q::qmatmul_cases()),
        of_cases("qconv", q::qconv_cases()),
        of_cases("resize", rs::resize_cases()),
        of_cases("attention", att::attention_cases()),
    ]
}
