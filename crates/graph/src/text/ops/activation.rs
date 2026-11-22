//! Text for activations: `relu`, `leaky_relu<alpha=0.01>`, `clip`, `prelu`.

use super::OpText;
use crate::ops::{Activation, F32Attr, GeluApproximation, Op};
use crate::text::attr::{AttrValue, Attrs, TextWord, text_words};
use crate::text::error::ParseError;

text_words! {
    GeluApproximation {
        GeluApproximation::None => "none",
        GeluApproximation::Tanh => "tanh",
    }
}

/// Activations without attributes.
const PLAIN: [Activation; 7] = [
    Activation::Relu,
    Activation::Sigmoid,
    Activation::Tanh,
    Activation::Softplus,
    Activation::Softsign,
    Activation::HardSwish,
    Activation::Mish,
];

const fn name(activation: Activation) -> &'static str {
    match activation {
        Activation::Relu => "relu",
        Activation::Sigmoid => "sigmoid",
        Activation::Tanh => "tanh",
        Activation::Softplus => "softplus",
        Activation::Softsign => "softsign",
        Activation::HardSwish => "hard_swish",
        Activation::Mish => "mish",
        Activation::Gelu(_) => "gelu",
        Activation::LeakyRelu { .. } => "leaky_relu",
        Activation::Elu { .. } => "elu",
        Activation::Celu { .. } => "celu",
        Activation::ThresholdedRelu { .. } => "thresholded_relu",
        Activation::Selu { .. } => "selu",
        Activation::HardSigmoid { .. } => "hard_sigmoid",
    }
}

fn float(value: F32Attr) -> AttrValue {
    AttrValue::float(value.get())
}

pub(super) fn encode(activation: Activation) -> OpText {
    let attrs = match activation {
        Activation::Gelu(approximation) => {
            vec![("approximate", AttrValue::word(approximation.word()))]
        }
        Activation::LeakyRelu { alpha }
        | Activation::Elu { alpha }
        | Activation::Celu { alpha }
        | Activation::ThresholdedRelu { alpha } => vec![("alpha", float(alpha))],
        Activation::Selu { alpha, gamma } => vec![("alpha", float(alpha)), ("gamma", float(gamma))],
        Activation::HardSigmoid { alpha, beta } => {
            vec![("alpha", float(alpha)), ("beta", float(beta))]
        }
        _ => Vec::new(),
    };
    OpText::new(name(activation), attrs)
}

pub(super) fn decode(op_name: &str, attrs: &mut Attrs) -> Option<Result<Op, ParseError>> {
    if let Some(&activation) = PLAIN.iter().find(|&&activation| name(activation) == op_name) {
        return Some(Ok(Op::Activation(activation)));
    }
    let decoded = match op_name {
        "clip" => return Some(Ok(Op::Clip)),
        "prelu" => return Some(Ok(Op::PRelu)),
        "leaky_relu" => take_alpha(attrs).map(|alpha| Activation::LeakyRelu { alpha }),
        "elu" => take_alpha(attrs).map(|alpha| Activation::Elu { alpha }),
        "celu" => take_alpha(attrs).map(|alpha| Activation::Celu { alpha }),
        "thresholded_relu" => take_alpha(attrs).map(|alpha| Activation::ThresholdedRelu { alpha }),
        "gelu" => {
            attrs.take("approximate").and_then(|value| value.enumerated()).map(Activation::Gelu)
        }
        "selu" => {
            decode_pair(attrs, "gamma").map(|(alpha, gamma)| Activation::Selu { alpha, gamma })
        }
        "hard_sigmoid" => {
            decode_pair(attrs, "beta").map(|(alpha, beta)| Activation::HardSigmoid { alpha, beta })
        }
        _ => return None,
    };
    Some(decoded.map(Op::Activation))
}

fn take_alpha(attrs: &mut Attrs) -> Result<F32Attr, ParseError> {
    attrs.take("alpha")?.f32_attr()
}

/// `alpha` and a second float attribute.
fn decode_pair(attrs: &mut Attrs, second: &'static str) -> Result<(F32Attr, F32Attr), ParseError> {
    Ok((take_alpha(attrs)?, attrs.take(second)?.f32_attr()?))
}
