//! Text for `attention`. `q_heads` and `kv_heads` appear only with the
//! packed layout, `qk` only when that output is produced, and `softcap`
//! only when it is set.

use super::OpText;
use crate::ops::{
    AttentionAttrs, AttentionLayout, AttentionMask, Causality, HeadCount, Op, Presence, QkOutput,
};
use crate::text::attr::{Attr, AttrValue, Attrs, TextWord, text_words};
use crate::text::error::ParseError;

text_words! {
    AttentionMask {
        AttentionMask::None => "none",
        AttentionMask::Boolean => "boolean",
        AttentionMask::Additive => "additive",
    }
    Causality {
        Causality::Full => "full",
        Causality::Causal => "causal",
    }
    Presence {
        Presence::Absent => "absent",
        Presence::Present => "present",
    }
    QkOutput {
        QkOutput::MatMul => "matmul",
        QkOutput::Softcapped => "softcapped",
        QkOutput::Masked => "masked",
        QkOutput::Softmax => "softmax",
    }
}

pub(super) fn encode(attrs: &AttentionAttrs) -> OpText {
    let mut fields = match attrs.layout {
        AttentionLayout::Heads => vec![("layout", AttrValue::word("heads"))],
        AttentionLayout::Packed { q_heads, kv_heads } => vec![
            ("layout", AttrValue::word("packed")),
            ("q_heads", AttrValue::int(q_heads.get())),
            ("kv_heads", AttrValue::int(kv_heads.get())),
        ],
    };
    fields.extend([
        ("mask", AttrValue::word(attrs.mask.word())),
        ("causality", AttrValue::word(attrs.causality.word())),
        ("past", AttrValue::word(attrs.past.word())),
        ("present", AttrValue::word(attrs.present.word())),
        ("nonpad_lengths", AttrValue::word(attrs.nonpad_lengths.word())),
    ]);
    if let Some(qk) = attrs.qk_output {
        fields.push(("qk", AttrValue::word(qk.word())));
    }
    fields.push(("scale", AttrValue::float(attrs.scale.get())));
    if let Some(softcap) = attrs.softcap {
        fields.push(("softcap", AttrValue::float(softcap.get())));
    }
    fields.push(("precision", AttrValue::word(attrs.softmax_precision.word())));
    OpText::new("attention", fields)
}

pub(super) fn decode(name: &str, attrs: &mut Attrs) -> Option<Result<Op, ParseError>> {
    (name == "attention").then(|| decode_attention(attrs))
}

fn decode_attention(attrs: &mut Attrs) -> Result<Op, ParseError> {
    let layout = decode_layout(attrs)?;
    Ok(Op::Attention(AttentionAttrs {
        layout,
        mask: attrs.take("mask")?.enumerated()?,
        causality: attrs.take("causality")?.enumerated()?,
        past: attrs.take("past")?.enumerated()?,
        present: attrs.take("present")?.enumerated()?,
        nonpad_lengths: attrs.take("nonpad_lengths")?.enumerated()?,
        qk_output: attrs.take_optional("qk").map(|qk| qk.enumerated()).transpose()?,
        scale: attrs.take("scale")?.f32_attr()?,
        softcap: attrs.take_optional("softcap").map(|softcap| softcap.f32_attr()).transpose()?,
        softmax_precision: attrs.take("precision")?.enumerated()?,
    }))
}

fn decode_layout(attrs: &mut Attrs) -> Result<AttentionLayout, ParseError> {
    let layout = attrs.take("layout")?;
    match layout.word()? {
        "heads" => Ok(AttentionLayout::Heads),
        "packed" => Ok(AttentionLayout::Packed {
            q_heads: heads(&attrs.take("q_heads")?)?,
            kv_heads: heads(&attrs.take("kv_heads")?)?,
        }),
        _ => Err(layout.bad("expected `heads` or `packed`")),
    }
}

fn heads(attr: &Attr) -> Result<HeadCount, ParseError> {
    HeadCount::new(attr.u64()?).ok_or_else(|| attr.bad("a head count must be positive"))
}
