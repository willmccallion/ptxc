//! ONNX `Attention`: layouts, masks, past state and the optional outputs.

use crate::builders::{
    BOOL, FLOAT, assert_lines, fixed, float, import, int, line_at, model_at, node,
};
use ptxc_onnx::Unsupported;
use ptxc_onnx_proto::{AttributeProto, ValueInfoProto};

fn heads() -> Vec<ValueInfoProto> {
    vec![
        fixed("q", FLOAT, &[1, 2, 3, 16]),
        fixed("k", FLOAT, &[1, 2, 5, 16]),
        fixed("v", FLOAT, &[1, 2, 5, 8]),
    ]
}

fn attention(
    names: &[&str],
    outputs: &[&str],
    inputs: Vec<ValueInfoProto>,
    attrs: Vec<AttributeProto>,
) -> String {
    line_at(23, node("Attention", names, outputs, attrs), inputs, vec![])
}

#[test]
fn layouts_masks_and_scaling() {
    let packed = vec![
        fixed("q", FLOAT, &[1, 3, 32]),
        fixed("k", FLOAT, &[1, 5, 16]),
        fixed("v", FLOAT, &[1, 5, 16]),
    ];
    let with_mask = |dtype| {
        let mut inputs = heads();
        inputs.push(fixed("m", dtype, &[3, 5]));
        attention(&["q", "k", "v", "m"], &["y"], inputs, vec![])
    };
    let attrs = vec![
        int("is_causal", 1),
        float("scale", 0.5),
        float("softcap", 30.0),
        int("softmax_precision", 1),
    ];

    assert_lines(&[
        (
            attention(&["q", "k", "v"], &["y"], heads(), vec![]),
            "attention<layout=heads, mask=none, causality=full, past=absent, present=absent, nonpad_lengths=absent, scale=0.25, precision=f32> %q, %k, %v : f32[1, 2, 3, 8]",
        ),
        (
            attention(
                &["q", "k", "v"],
                &["y"],
                packed,
                vec![int("q_num_heads", 2), int("kv_num_heads", 1)],
            ),
            "attention<layout=packed, q_heads=2, kv_heads=1, mask=none, causality=full, past=absent, present=absent, nonpad_lengths=absent, scale=0.25, precision=f32> %q, %k, %v : f32[1, 3, 32]",
        ),
        (
            with_mask(BOOL),
            "attention<layout=heads, mask=boolean, causality=full, past=absent, present=absent, nonpad_lengths=absent, scale=0.25, precision=f32> %q, %k, %v, %m : f32[1, 2, 3, 8]",
        ),
        (
            with_mask(FLOAT),
            "attention<layout=heads, mask=additive, causality=full, past=absent, present=absent, nonpad_lengths=absent, scale=0.25, precision=f32> %q, %k, %v, %m : f32[1, 2, 3, 8]",
        ),
        (
            attention(&["q", "k", "v"], &["y"], heads(), attrs),
            "attention<layout=heads, mask=none, causality=causal, past=absent, present=absent, nonpad_lengths=absent, scale=0.5, softcap=30.0, precision=f32> %q, %k, %v : f32[1, 2, 3, 8]",
        ),
    ]);
}

#[test]
fn past_state_and_optional_outputs() {
    let mut with_past = heads();
    with_past.extend([fixed("pk", FLOAT, &[1, 2, 4, 16]), fixed("pv", FLOAT, &[1, 2, 4, 8])]);
    let qk = |mode| {
        attention(
            &["q", "k", "v"],
            &["y", "", "", "qk"],
            heads(),
            vec![int("qk_matmul_output_mode", mode)],
        )
    };

    assert_lines(&[
        (
            attention(
                &["q", "k", "v", "", "pk", "pv"],
                &["y", "pk_out", "pv_out"],
                with_past,
                vec![],
            ),
            "attention<layout=heads, mask=none, causality=full, past=present, present=present, nonpad_lengths=absent, scale=0.25, precision=f32> %q, %k, %v, %pk, %pv : f32[1, 2, 3, 8], f32[1, 2, 9, 16], f32[1, 2, 9, 8]",
        ),
        (
            qk(0),
            "attention<layout=heads, mask=none, causality=full, past=absent, present=present, nonpad_lengths=absent, qk=matmul, scale=0.25, precision=f32> %q, %k, %v : f32[1, 2, 3, 8], f32[1, 2, 5, 16], f32[1, 2, 5, 8], f32[1, 2, 3, 5]",
        ),
        (
            qk(1),
            "attention<layout=heads, mask=none, causality=full, past=absent, present=present, nonpad_lengths=absent, qk=softcapped, scale=0.25, precision=f32> %q, %k, %v : f32[1, 2, 3, 8], f32[1, 2, 5, 16], f32[1, 2, 5, 8], f32[1, 2, 3, 5]",
        ),
        (
            qk(2),
            "attention<layout=heads, mask=none, causality=full, past=absent, present=present, nonpad_lengths=absent, qk=masked, scale=0.25, precision=f32> %q, %k, %v : f32[1, 2, 3, 8], f32[1, 2, 5, 16], f32[1, 2, 5, 8], f32[1, 2, 3, 5]",
        ),
        (
            qk(3),
            "attention<layout=heads, mask=none, causality=full, past=absent, present=present, nonpad_lengths=absent, qk=softmax, scale=0.25, precision=f32> %q, %k, %v : f32[1, 2, 3, 8], f32[1, 2, 5, 16], f32[1, 2, 5, 8], f32[1, 2, 3, 5]",
        ),
    ]);
}

#[test]
fn a_past_key_without_a_past_value_is_unsupported() {
    let mut inputs = heads();
    inputs.push(fixed("pk", FLOAT, &[1, 2, 4, 16]));
    let model = model_at(
        23,
        vec![node("Attention", &["q", "k", "v", "", "pk"], &["y"], vec![])],
        inputs,
        vec![],
        vec![],
    );

    assert_eq!(
        import(&model).unwrap_err().unsupported(),
        Some(&Unsupported::Attribute {
            op: "Attention".to_owned(),
            attribute: "past_key without past_value".to_owned()
        })
    );
}
