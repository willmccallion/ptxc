//! Text for elementwise ops: `abs`, `add`, `mod<kind=fmod>`, `cast<to=f16>`.

use super::OpText;
use crate::ops::{BinaryOp, CastAttrs, InfDetection, ModKind, Op, UnaryOp, VariadicOp};
use crate::text::attr::{AttrValue, Attrs, TextWord, text_words};
use crate::text::error::ParseError;
use ptxc_core::DType;
use ptxc_core::float::Overflow;

text_words! {
    InfDetection {
        InfDetection::Both => "both",
        InfDetection::Positive => "positive",
        InfDetection::Negative => "negative",
        InfDetection::Neither => "neither",
    }
    ModKind {
        ModKind::Integer => "integer",
        ModKind::Fmod => "fmod",
    }
    Overflow {
        Overflow::Saturate => "saturate",
        Overflow::NonFinite => "non_finite",
    }
    VariadicOp {
        VariadicOp::Max => "max",
        VariadicOp::Min => "min",
        VariadicOp::Sum => "sum",
        VariadicOp::Mean => "mean",
    }
}

/// Unary ops without attributes; `IsInf` is read separately.
const PLAIN_UNARY: [UnaryOp; 15] = [
    UnaryOp::Abs,
    UnaryOp::Neg,
    UnaryOp::Exp,
    UnaryOp::Log,
    UnaryOp::Sqrt,
    UnaryOp::Reciprocal,
    UnaryOp::Floor,
    UnaryOp::Ceil,
    UnaryOp::Round,
    UnaryOp::Sign,
    UnaryOp::Erf,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Not,
    UnaryOp::IsNaN,
];

/// Binary ops without attributes; `Mod` is read separately.
const PLAIN_BINARY: [BinaryOp; 13] = [
    BinaryOp::Add,
    BinaryOp::Sub,
    BinaryOp::Mul,
    BinaryOp::Div,
    BinaryOp::Pow,
    BinaryOp::Equal,
    BinaryOp::Less,
    BinaryOp::LessOrEqual,
    BinaryOp::Greater,
    BinaryOp::GreaterOrEqual,
    BinaryOp::And,
    BinaryOp::Or,
    BinaryOp::Xor,
];

const fn unary_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Abs => "abs",
        UnaryOp::Neg => "neg",
        UnaryOp::Exp => "exp",
        UnaryOp::Log => "log",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Reciprocal => "reciprocal",
        UnaryOp::Floor => "floor",
        UnaryOp::Ceil => "ceil",
        UnaryOp::Round => "round",
        UnaryOp::Sign => "sign",
        UnaryOp::Erf => "erf",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
        UnaryOp::Not => "not",
        UnaryOp::IsNaN => "is_nan",
        UnaryOp::IsInf(_) => "is_inf",
    }
}

const fn binary_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Pow => "pow",
        BinaryOp::Mod(_) => "mod",
        BinaryOp::Equal => "equal",
        BinaryOp::Less => "less",
        BinaryOp::LessOrEqual => "less_or_equal",
        BinaryOp::Greater => "greater",
        BinaryOp::GreaterOrEqual => "greater_or_equal",
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
        BinaryOp::Xor => "xor",
    }
}

pub(super) fn encode_unary(op: UnaryOp) -> OpText {
    match op {
        UnaryOp::IsInf(detect) => {
            OpText::new("is_inf", vec![("detect", AttrValue::word(detect.word()))])
        }
        _ => OpText::plain(unary_name(op)),
    }
}

pub(super) fn encode_binary(op: BinaryOp) -> OpText {
    match op {
        BinaryOp::Mod(kind) => OpText::new("mod", vec![("kind", AttrValue::word(kind.word()))]),
        _ => OpText::plain(binary_name(op)),
    }
}

pub(super) fn encode_variadic(op: VariadicOp) -> OpText {
    OpText::plain(op.word())
}

pub(super) fn encode_cast(attrs: CastAttrs) -> OpText {
    let mut fields = vec![("to", AttrValue::word(attrs.to().word()))];
    if matches!(attrs.to(), DType::F8E4M3Fn | DType::F8E5M2) {
        fields.push(("overflow", AttrValue::word(attrs.overflow().word())));
    }
    OpText::new("cast", fields)
}

pub(super) fn decode(name: &str, attrs: &mut Attrs) -> Option<Result<Op, ParseError>> {
    if let Some(&op) = PLAIN_UNARY.iter().find(|&&op| unary_name(op) == name) {
        return Some(Ok(Op::Unary(op)));
    }
    if let Some(&op) = PLAIN_BINARY.iter().find(|&&op| binary_name(op) == name) {
        return Some(Ok(Op::Binary(op)));
    }
    if let Some(op) = VariadicOp::from_word(name) {
        return Some(Ok(Op::Variadic(op)));
    }
    Some(match name {
        "is_inf" => attrs
            .take("detect")
            .and_then(|detect| detect.enumerated())
            .map(|d| Op::Unary(UnaryOp::IsInf(d))),
        "mod" => attrs
            .take("kind")
            .and_then(|kind| kind.enumerated())
            .map(|k| Op::Binary(BinaryOp::Mod(k))),
        "where" => Ok(Op::Where),
        "cast" => decode_cast(attrs),
        _ => return None,
    })
}

/// `overflow` is written only for FP8 targets, where it matters.
fn decode_cast(attrs: &mut Attrs) -> Result<Op, ParseError> {
    let to = attrs.take("to")?.enumerated()?;
    let overflow = match attrs.take_optional("overflow") {
        Some(overflow) => overflow.enumerated()?,
        None => Overflow::Saturate,
    };
    Ok(Op::Cast(CastAttrs::new(to, overflow)))
}
