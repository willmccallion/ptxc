//! Text for softmax and normalisation: `softmax<axis=1>`,
//! `layer_norm<axis=2, epsilon=1e-5, stash=f32, outputs=normalized>`.

use super::OpText;
use crate::ops::{
    BatchNormAttrs, F32Attr, InstanceNormAttrs, LayerNormAttrs, LayerNormKind, NormOutputs, Op,
    SoftmaxAttrs, SoftmaxKind, StashType,
};
use crate::text::attr::{AttrValue, Attrs, TextWord, text_words};
use crate::text::error::ParseError;

text_words! {
    SoftmaxKind {
        SoftmaxKind::Softmax => "softmax",
        SoftmaxKind::LogSoftmax => "log_softmax",
    }
    StashType {
        StashType::F32 => "f32",
        StashType::Bf16 => "bf16",
    }
    NormOutputs {
        NormOutputs::Normalized => "normalized",
        NormOutputs::WithStatistics => "with_statistics",
    }
}

pub(super) fn encode_softmax(attrs: SoftmaxAttrs) -> OpText {
    OpText::new(attrs.kind.word(), vec![("axis", AttrValue::int(attrs.axis.index()))])
}

pub(super) fn encode_layer_norm(attrs: LayerNormAttrs) -> OpText {
    let mut fields = vec![
        ("axis", AttrValue::int(attrs.axis.index())),
        ("epsilon", AttrValue::float(attrs.epsilon.get())),
        ("stash", AttrValue::word(attrs.stash.word())),
    ];
    match attrs.kind {
        LayerNormKind::Layer(outputs) => {
            fields.push(("outputs", AttrValue::word(outputs.word())));
            OpText::new("layer_norm", fields)
        }
        LayerNormKind::Rms => OpText::new("rms_norm", fields),
    }
}

pub(super) fn encode_epsilon(name: &'static str, epsilon: F32Attr) -> OpText {
    OpText::new(name, vec![("epsilon", AttrValue::float(epsilon.get()))])
}

pub(super) fn decode(name: &str, attrs: &mut Attrs) -> Option<Result<Op, ParseError>> {
    if let Some(kind) = SoftmaxKind::from_word(name) {
        return Some(
            attrs
                .take("axis")
                .and_then(|axis| axis.axis())
                .map(|axis| Op::Softmax(SoftmaxAttrs { kind, axis })),
        );
    }
    Some(match name {
        "layer_norm" => attrs
            .take("outputs")
            .and_then(|outputs| outputs.enumerated())
            .and_then(|o| decode_layer_norm(attrs, LayerNormKind::Layer(o))),
        "rms_norm" => decode_layer_norm(attrs, LayerNormKind::Rms),
        "batch_norm" => epsilon(attrs).map(|epsilon| Op::BatchNorm(BatchNormAttrs { epsilon })),
        "instance_norm" => {
            epsilon(attrs).map(|epsilon| Op::InstanceNorm(InstanceNormAttrs { epsilon }))
        }
        _ => return None,
    })
}

fn decode_layer_norm(attrs: &mut Attrs, kind: LayerNormKind) -> Result<Op, ParseError> {
    Ok(Op::LayerNorm(LayerNormAttrs {
        kind,
        axis: attrs.take("axis")?.axis()?,
        epsilon: epsilon(attrs)?,
        stash: attrs.take("stash")?.enumerated()?,
    }))
}

fn epsilon(attrs: &mut Attrs) -> Result<F32Attr, ParseError> {
    attrs.take("epsilon")?.f32_attr()
}
