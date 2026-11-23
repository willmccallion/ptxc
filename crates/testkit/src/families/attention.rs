//! `Attention` in both layouts with grouped key and value heads, every mask
//! kind (a short last axis included), causality, a past, non-pad lengths,
//! softcapping and every QK output.

use super::values::{Domain, tensor};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::select;
use ptxc_core::{DType, HostTensor};
use ptxc_graph::{
    AttentionAttrs, AttentionLayout, AttentionMask, Causality, F32Attr, HeadCount, Layout, Op,
    Presence, QkOutput, TensorType, TypeRule,
};

const FLOATS: [DType; 3] = [DType::F32, DType::F16, DType::Bf16];

/// Attention's sizes: batch, query and key heads, query and key lengths,
/// past length, and head sizes for keys and values.
#[derive(Clone, Copy, Debug)]
struct Sizes {
    batch: u64,
    q_heads: u64,
    kv_heads: u64,
    q_sequence: u64,
    kv_sequence: u64,
    past: u64,
    head_size: u64,
    v_head_size: u64,
}

fn sizes() -> impl Strategy<Value = Sizes> {
    (
        1_u64..=2,
        select(vec![(1_u64, 1_u64), (2, 2), (2, 1), (4, 2)]),
        1_u64..=3,
        0_u64..=3,
        0_u64..=2,
        1_u64..=3,
        1_u64..=3,
    )
        .prop_map(
            |(
                batch,
                (q_heads, kv_heads),
                q_sequence,
                kv_sequence,
                past,
                head_size,
                v_head_size,
            )| Sizes {
                batch,
                q_heads,
                kv_heads,
                q_sequence,
                kv_sequence,
                past,
                head_size,
                v_head_size,
            },
        )
}

fn layout(sizes: Sizes) -> impl Strategy<Value = AttentionLayout> {
    let packed = AttentionLayout::Packed {
        q_heads: HeadCount::new(sizes.q_heads).unwrap(),
        kv_heads: HeadCount::new(sizes.kv_heads).unwrap(),
    };
    prop_oneof![Just(AttentionLayout::Heads), Just(packed)]
}

/// An operand of `heads` heads of `length` positions of `size` elements.
fn heads_dims(
    layout: AttentionLayout,
    batch: u64,
    (heads, length, size): (u64, u64, u64),
) -> Vec<u64> {
    match layout {
        AttentionLayout::Heads => vec![batch, heads, length, size],
        AttentionLayout::Packed { .. } => vec![batch, length, heads * size],
    }
}

/// A mask's dims: broadcastable to `(batch, heads, queries, keys)`, its
/// last axis from 1 to every key.
fn mask_dims(sizes: Sizes) -> impl Strategy<Value = Vec<u64>> {
    let total = sizes.past + sizes.kv_sequence;
    let leading = [sizes.batch, sizes.q_heads, sizes.q_sequence];
    (0_usize..=3, vec(any::<bool>(), 3), 0..=total).prop_map(move |(dropped, ones, last)| {
        let kept = leading[dropped..]
            .iter()
            .zip(&ones[dropped..])
            .map(|(&d, &one)| if one { 1 } else { d });
        kept.chain([last]).collect()
    })
}

fn qk_output() -> impl Strategy<Value = Option<QkOutput>> {
    prop_oneof![
        Just(None),
        Just(Some(QkOutput::MatMul)),
        Just(Some(QkOutput::Softcapped)),
        Just(Some(QkOutput::Masked)),
        Just(Some(QkOutput::Softmax)),
    ]
}

/// `Attention` over head layouts, masks, caches and its optional QK
/// output, every case one ONNX admits.
pub fn attention_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let mask_kind = prop_oneof![
        Just(AttentionMask::None),
        Just(AttentionMask::Boolean),
        Just(AttentionMask::Additive)
    ];
    let flags = any::<[bool; 4]>();
    (select(FLOATS.to_vec()), sizes(), mask_kind, flags, qk_output())
        .prop_flat_map(|(dtype, sizes, mask, [causal, past, present, lengths], qk_output)| {
            // ONNX allows non-pad lengths only without a key-value cache.
            let lengths = lengths && !past && !present;
            let sizes = Sizes { past: if past { sizes.past } else { 0 }, ..sizes };
            // A softcap of 0 is ONNX's way of leaving it off.
            let softcap = select(vec![None, Some(F32Attr::new(2.0)), Some(F32Attr::new(0.0))]);
            (layout(sizes), mask_dims(sizes), softcap).prop_flat_map(
                move |(layout, mask_shape, softcap)| {
                    let attrs = AttentionAttrs {
                        layout,
                        mask,
                        causality: if causal { Causality::Causal } else { Causality::Full },
                        past: if past { Presence::Present } else { Presence::Absent },
                        present: if present { Presence::Present } else { Presence::Absent },
                        nonpad_lengths: if lengths { Presence::Present } else { Presence::Absent },
                        qk_output,
                        scale: F32Attr::new(0.5),
                        softcap,
                        softmax_precision: DType::F32,
                    };
                    inputs(attrs, sizes, dtype, mask_shape)
                        .prop_map(move |inputs| (Op::Attention(attrs), inputs))
                },
            )
        })
        .prop_map(|(op, inputs)| {
            assert!(admitted(&op, &inputs), "the type rule rejects {op:?}");
            (op, inputs)
        })
}

/// Q, K and V, then the operands `attrs` says are present.
fn inputs(
    attrs: AttentionAttrs,
    sizes: Sizes,
    dtype: DType,
    mask_shape: Vec<u64>,
) -> BoxedStrategy<Vec<HostTensor>> {
    let Sizes { batch, q_heads, kv_heads, q_sequence, kv_sequence, past, head_size, v_head_size } =
        sizes;
    let operand = |dims| tensor(dtype, dims, Domain::Finite);
    let mut parts = vec![
        operand(heads_dims(attrs.layout, batch, (q_heads, q_sequence, head_size))),
        operand(heads_dims(attrs.layout, batch, (kv_heads, kv_sequence, head_size))),
        operand(heads_dims(attrs.layout, batch, (kv_heads, kv_sequence, v_head_size))),
    ];
    match attrs.mask {
        AttentionMask::None => {}
        AttentionMask::Boolean => parts.push(tensor(DType::Bool, mask_shape, Domain::Any)),
        AttentionMask::Additive => parts.push(operand(mask_shape)),
    }
    if attrs.past == Presence::Present {
        parts.push(operand(vec![batch, kv_heads, past, head_size]));
        parts.push(operand(vec![batch, kv_heads, past, v_head_size]));
    }
    if attrs.nonpad_lengths == Presence::Present {
        let total = i64::try_from(past + kv_sequence).unwrap();
        let lengths = vec(0..=total, usize::try_from(batch).unwrap())
            .prop_map(|lengths| HostTensor::vector(lengths.into()));
        parts.push(lengths.boxed());
    }
    parts.boxed()
}

fn admitted(op: &Op, inputs: &[HostTensor]) -> bool {
    let types: Vec<TensorType> = inputs
        .iter()
        .map(|input| TensorType::new(input.dtype(), input.shape().clone(), Layout::RowMajor))
        .collect();
    op.infer(&types.iter().collect::<Vec<_>>()).is_ok()
}
