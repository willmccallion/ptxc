//! Text for `matmul` and `gemm<a=as_is, b=transposed, alpha=1.0, beta=1.0>`.

use super::OpText;
use crate::ops::{GemmAttrs, Op, Orientation};
use crate::text::attr::{AttrValue, Attrs, TextWord, text_words};
use crate::text::error::ParseError;

text_words! {
    Orientation {
        Orientation::AsIs => "as_is",
        Orientation::Transposed => "transposed",
    }
}

pub(super) fn encode_gemm(attrs: GemmAttrs) -> OpText {
    OpText::new(
        "gemm",
        vec![
            ("a", AttrValue::word(attrs.a.word())),
            ("b", AttrValue::word(attrs.b.word())),
            ("alpha", AttrValue::float(attrs.alpha.get())),
            ("beta", AttrValue::float(attrs.beta.get())),
        ],
    )
}

pub(super) fn decode(name: &str, attrs: &mut Attrs) -> Option<Result<Op, ParseError>> {
    match name {
        "matmul" => Some(Ok(Op::MatMul)),
        "gemm" => Some(decode_gemm(attrs)),
        _ => None,
    }
}

fn decode_gemm(attrs: &mut Attrs) -> Result<Op, ParseError> {
    Ok(Op::Gemm(GemmAttrs {
        a: attrs.take("a")?.enumerated()?,
        b: attrs.take("b")?.enumerated()?,
        alpha: attrs.take("alpha")?.f32_attr()?,
        beta: attrs.take("beta")?.f32_attr()?,
    }))
}
