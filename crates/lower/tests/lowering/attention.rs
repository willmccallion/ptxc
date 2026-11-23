use crate::{stages, tensor_type};
use ptxc_core::DType;
use ptxc_graph::{
    AttentionAttrs, AttentionLayout, AttentionMask, Causality, F32Attr, GraphBuilder, HeadCount,
    Op, Presence, QkOutput,
};
use ptxc_loop::print_compute;
use ptxc_testkit::golden;

const fn attrs(layout: AttentionLayout) -> AttentionAttrs {
    AttentionAttrs {
        layout,
        mask: AttentionMask::None,
        causality: Causality::Full,
        past: Presence::Absent,
        present: Presence::Absent,
        nonpad_lengths: Presence::Absent,
        qk_output: None,
        scale: F32Attr::new(0.5),
        softcap: None,
        softmax_precision: DType::F32,
    }
}

/// Every stage of a graph whose one node is `attention` on inputs typed
/// `operands`, every output read.
fn lowered(attention: AttentionAttrs, operands: &[(DType, &[u64])]) -> String {
    let mut builder = GraphBuilder::new();
    let inputs =
        operands.iter().map(|&(dtype, dims)| builder.add_input(tensor_type(dtype, dims))).collect();
    let outputs = builder.add_node(Op::Attention(attention), inputs).unwrap();
    let graph = builder.finish(outputs).unwrap();
    stages(graph).unwrap().iter().map(|stage| print_compute(stage.compute())).collect()
}

#[test]
fn attention_scores_then_normalises_then_weighs_values_with_grouped_heads_after_a_past() {
    let causal = AttentionAttrs {
        mask: AttentionMask::Boolean,
        causality: Causality::Causal,
        past: Presence::Present,
        softcap: Some(F32Attr::new(2.0)),
        ..attrs(AttentionLayout::Heads)
    };
    let definitions = lowered(
        causal,
        &[
            (DType::F16, &[1, 2, 2, 3]),
            (DType::F16, &[1, 1, 2, 3]),
            (DType::F16, &[1, 1, 2, 4]),
            (DType::Bool, &[2, 3]),
            (DType::F16, &[1, 1, 1, 3]),
            (DType::F16, &[1, 1, 1, 4]),
        ],
    );

    golden::check("attention/causal_past.txt", &definitions).unwrap();
}

#[test]
fn packed_attention_writes_present_keys_values_and_the_softmax() {
    let packed = AttentionLayout::Packed {
        q_heads: HeadCount::new(2).unwrap(),
        kv_heads: HeadCount::new(2).unwrap(),
    };
    let with_outputs = AttentionAttrs {
        present: Presence::Present,
        qk_output: Some(QkOutput::Softmax),
        ..attrs(packed)
    };
    let definitions = lowered(
        with_outputs,
        &[(DType::F32, &[1, 2, 4]), (DType::F32, &[1, 3, 4]), (DType::F32, &[1, 3, 2])],
    );

    golden::check("attention/packed_outputs.txt", &definitions).unwrap();
}
