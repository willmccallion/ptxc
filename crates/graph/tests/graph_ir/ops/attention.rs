use crate::common::{f32_type, tensor_type};
use ptxc_core::{Axis, DType, Dim, Rank};
use ptxc_graph::{
    AttentionAttrs, AttentionLayout, AttentionMask, Causality, F32Attr, HeadCount, Presence,
    QkOutput, TensorType, TypeError, TypeRule,
};

const PLAIN: AttentionAttrs = AttentionAttrs {
    layout: AttentionLayout::Heads,
    mask: AttentionMask::None,
    causality: Causality::Full,
    past: Presence::Absent,
    present: Presence::Absent,
    nonpad_lengths: Presence::Absent,
    qk_output: None,
    scale: F32Attr::new(0.25),
    softcap: None,
    softmax_precision: DType::F32,
};

fn axis(index: i64) -> Axis {
    Axis::from_signed(index, Rank::new(4)).unwrap()
}

/// Grouped-query Q, K and V: 8 query heads over 2 key/value heads.
fn qkv() -> [TensorType; 3] {
    [f32_type(&[2, 8, 5, 16]), f32_type(&[2, 2, 7, 16]), f32_type(&[2, 2, 7, 32])]
}

fn infer(attrs: &AttentionAttrs, operands: &[TensorType]) -> Result<Vec<TensorType>, TypeError> {
    attrs.infer(&operands.iter().collect::<Vec<_>>())
}

fn with(extra: &[TensorType]) -> Vec<TensorType> {
    qkv().into_iter().chain(extra.iter().cloned()).collect()
}

#[test]
fn grouped_heads_give_one_output_per_query_head() {
    assert_eq!(infer(&PLAIN, &qkv()), Ok(vec![f32_type(&[2, 8, 5, 32])]));
}

#[test]
fn key_heads_must_divide_query_heads() {
    let operands = [f32_type(&[2, 6, 5, 16]), f32_type(&[2, 4, 7, 16]), f32_type(&[2, 4, 7, 32])];
    let empty = [f32_type(&[2, 6, 5, 16]), f32_type(&[2, 0, 7, 16]), f32_type(&[2, 0, 7, 32])];

    assert_eq!(
        infer(&PLAIN, &operands),
        Err(TypeError::HeadsNotDivisible { q_heads: Dim::new(6), kv_heads: Dim::new(4) })
    );
    assert!(matches!(infer(&PLAIN, &empty), Err(TypeError::HeadsNotDivisible { .. })));
}

#[test]
fn key_and_value_must_agree_with_query() {
    let [q, _, v] = qkv();

    let error = infer(&PLAIN, &[q.clone(), f32_type(&[2, 2, 7, 8]), v]);

    assert_eq!(
        error,
        Err(TypeError::DimMismatch { axis: axis(3), expected: Dim::new(16), actual: Dim::new(8) })
    );
    let [_, k, _] = qkv();
    assert!(matches!(
        infer(&PLAIN, &[q, k, f32_type(&[2, 2, 6, 32])]),
        Err(TypeError::DimMismatch { .. })
    ));
}

#[test]
fn the_heads_layout_needs_4_d_operands() {
    let [_, k, v] = qkv();

    assert!(matches!(infer(&PLAIN, &[f32_type(&[2, 5, 16]), k, v]), Err(TypeError::Rank { .. })));
}

#[test]
fn the_packed_layout_splits_hidden_axes_into_heads() {
    let packed = AttentionLayout::Packed {
        q_heads: HeadCount::new(8).unwrap(),
        kv_heads: HeadCount::new(2).unwrap(),
    };
    let attrs = AttentionAttrs { layout: packed, ..PLAIN };
    let operands = [f32_type(&[2, 5, 128]), f32_type(&[2, 7, 32]), f32_type(&[2, 7, 64])];

    assert_eq!(infer(&attrs, &operands), Ok(vec![f32_type(&[2, 5, 256])]));
    let uneven = [f32_type(&[2, 5, 130]), f32_type(&[2, 7, 32]), f32_type(&[2, 7, 64])];
    assert_eq!(
        infer(&attrs, &uneven),
        Err(TypeError::HiddenNotDivisible {
            hidden: Dim::new(130),
            heads: HeadCount::new(8).unwrap()
        })
    );
    let wide_key = [f32_type(&[2, 5, 128]), f32_type(&[2, 7, 48]), f32_type(&[2, 7, 64])];
    assert!(matches!(infer(&attrs, &wide_key), Err(TypeError::DimMismatch { .. })));
}

#[test]
fn masks_broadcast_to_the_scores_and_may_be_short() {
    let boolean = AttentionAttrs { mask: AttentionMask::Boolean, ..PLAIN };
    let additive = AttentionAttrs { mask: AttentionMask::Additive, ..PLAIN };

    assert!(infer(&boolean, &with(&[tensor_type(DType::Bool, &[5, 7])])).is_ok());
    assert!(infer(&additive, &with(&[f32_type(&[1, 1, 5, 3])])).is_ok());
    assert!(matches!(
        infer(&additive, &with(&[f32_type(&[5, 9])])),
        Err(TypeError::NotUnidirectional { .. })
    ));
}

#[test]
fn mask_types_follow_their_kind() {
    let boolean = AttentionAttrs { mask: AttentionMask::Boolean, ..PLAIN };
    let additive = AttentionAttrs { mask: AttentionMask::Additive, ..PLAIN };

    assert_eq!(
        infer(&boolean, &with(&[f32_type(&[5, 7])])),
        Err(TypeError::MixedDTypes { lhs: DType::Bool, rhs: DType::F32 })
    );
    assert_eq!(
        infer(&additive, &with(&[tensor_type(DType::F16, &[5, 7])])),
        Err(TypeError::MixedDTypes { lhs: DType::F32, rhs: DType::F16 })
    );
}

#[test]
fn a_kv_cache_extends_the_key_sequence() {
    let attrs = AttentionAttrs {
        mask: AttentionMask::Additive,
        past: Presence::Present,
        present: Presence::Present,
        qk_output: Some(QkOutput::Softmax),
        ..PLAIN
    };
    let operands = with(&[f32_type(&[5, 10]), f32_type(&[2, 2, 3, 16]), f32_type(&[2, 2, 3, 32])]);

    let expected = vec![
        f32_type(&[2, 8, 5, 32]),
        f32_type(&[2, 2, 10, 16]),
        f32_type(&[2, 2, 10, 32]),
        f32_type(&[2, 8, 5, 10]),
    ];
    assert_eq!(infer(&attrs, &operands), Ok(expected));
}

#[test]
fn the_present_cache_without_a_past_is_the_new_keys() {
    let attrs = AttentionAttrs { present: Presence::Present, ..PLAIN };

    let outputs = infer(&attrs, &qkv()).unwrap();

    assert_eq!(outputs[1..], [f32_type(&[2, 2, 7, 16]), f32_type(&[2, 2, 7, 32])]);
}

#[test]
fn the_past_cache_must_match_key_and_value() {
    let attrs = AttentionAttrs { past: Presence::Present, ..PLAIN };

    let bad_head = with(&[f32_type(&[2, 2, 3, 8]), f32_type(&[2, 2, 3, 32])]);
    let bad_length = with(&[f32_type(&[2, 2, 3, 16]), f32_type(&[2, 2, 4, 32])]);

    assert!(matches!(infer(&attrs, &bad_head), Err(TypeError::DimMismatch { .. })));
    assert!(matches!(infer(&attrs, &bad_length), Err(TypeError::DimMismatch { .. })));
}

#[test]
fn the_operand_count_follows_the_attributes() {
    let attrs = AttentionAttrs { mask: AttentionMask::Boolean, ..PLAIN };

    let error = infer(&attrs, &qkv()).unwrap_err();

    assert!(
        matches!(error, TypeError::Arity { expected, actual } if expected.get() == 4 && actual.get() == 3)
    );
}

#[test]
fn nonpad_lengths_are_i64_per_sample_and_exclude_a_cache() {
    let attrs = AttentionAttrs { nonpad_lengths: Presence::Present, ..PLAIN };

    assert!(infer(&attrs, &with(&[tensor_type(DType::I64, &[2])])).is_ok());
    assert!(infer(&attrs, &with(&[tensor_type(DType::I64, &[3])])).is_err());
    assert!(infer(&attrs, &with(&[tensor_type(DType::I32, &[2])])).is_err());
    let cached = AttentionAttrs { present: Presence::Present, ..attrs };
    assert_eq!(
        infer(&cached, &with(&[tensor_type(DType::I64, &[2])])),
        Err(TypeError::NonpadWithCache)
    );
}

#[test]
fn value_may_have_its_own_float_type() {
    let attrs = AttentionAttrs { present: Presence::Present, ..PLAIN };
    let [q, k, _] = qkv();

    let outputs = infer(&attrs, &[q, k, tensor_type(DType::F16, &[2, 2, 7, 32])]).unwrap();

    assert_eq!(
        outputs.iter().map(TensorType::dtype).collect::<Vec<_>>(),
        [DType::F32, DType::F32, DType::F16]
    );
}

#[test]
fn query_key_and_precision_must_be_floats_of_one_type() {
    let [q, _, v] = qkv();

    assert!(matches!(
        infer(&PLAIN, &[q, tensor_type(DType::F16, &[2, 2, 7, 16]), v.clone()]),
        Err(TypeError::MixedDTypes { .. })
    ));
    let integer = AttentionAttrs { softmax_precision: DType::I32, ..PLAIN };
    assert_eq!(infer(&integer, &qkv()), Err(TypeError::UnsupportedDType(DType::I32)));
    let [_, k, _] = qkv();
    assert_eq!(
        infer(&PLAIN, &[tensor_type(DType::I8, &[2, 8, 5, 16]), k, v]),
        Err(TypeError::UnsupportedDType(DType::I8))
    );
    assert_eq!(HeadCount::new(0), None);
}
