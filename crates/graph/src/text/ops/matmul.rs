//! Text for `matmul`, `gemm<a=as_is, b=transposed, alpha=1.0, beta=1.0>`
//! and `matmul_integer<a=[3], b=[1, [0, 2, 5]]>`: zero points are `[point]`
//! per tensor or `[axis, [point, ...]]` per axis.

use super::OpText;
use crate::ops::{GemmAttrs, MatMulIntegerAttrs, Op, Orientation, ZeroPoints};
use crate::text::attr::{Attr, AttrValue, Attrs, TextWord, text_words};
use crate::text::error::ParseError;
use ptxc_core::ZeroPoint;

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

fn zero_points_value(points: &ZeroPoints) -> AttrValue {
    let point = |point: &ZeroPoint| AttrValue::int(point.get());
    match points {
        ZeroPoints::PerTensor(only) => AttrValue::List(vec![point(only)]),
        ZeroPoints::PerAxis { axis, points } => {
            AttrValue::List(vec![AttrValue::int(axis.index()), AttrValue::list(points, point)])
        }
    }
}

pub(super) fn encode_matmul_integer(attrs: &MatMulIntegerAttrs) -> OpText {
    OpText::new(
        "matmul_integer",
        vec![("a", zero_points_value(&attrs.a)), ("b", zero_points_value(&attrs.b))],
    )
}

pub(super) fn decode(name: &str, attrs: &mut Attrs) -> Option<Result<Op, ParseError>> {
    match name {
        "matmul" => Some(Ok(Op::MatMul)),
        "gemm" => Some(decode_gemm(attrs)),
        "matmul_integer" => Some(decode_matmul_integer(attrs)),
        _ => None,
    }
}

fn decode_matmul_integer(attrs: &mut Attrs) -> Result<Op, ParseError> {
    Ok(Op::MatMulInteger(MatMulIntegerAttrs {
        a: decode_zero_points(&attrs.take("a")?)?,
        b: decode_zero_points(&attrs.take("b")?)?,
    }))
}

fn decode_zero_points(attr: &Attr) -> Result<ZeroPoints, ParseError> {
    match attr.items()?.as_slice() {
        [point] => Ok(ZeroPoints::PerTensor(decode_zero_point(point)?)),
        [axis, points] => Ok(ZeroPoints::PerAxis {
            axis: axis.axis()?,
            points: points.items()?.iter().map(decode_zero_point).collect::<Result<_, _>>()?,
        }),
        _ => Err(attr.bad("expected [point] or [axis, points]")),
    }
}

fn decode_zero_point(attr: &Attr) -> Result<ZeroPoint, ParseError> {
    let value = i32::try_from(attr.i64()?).map_err(|_| attr.bad("a zero point must fit in i32"))?;
    Ok(ZeroPoint::new(value))
}

fn decode_gemm(attrs: &mut Attrs) -> Result<Op, ParseError> {
    Ok(Op::Gemm(GemmAttrs {
        a: attrs.take("a")?.enumerated()?,
        b: attrs.take("b")?.enumerated()?,
        alpha: attrs.take("alpha")?.f32_attr()?,
        beta: attrs.take("beta")?.f32_attr()?,
    }))
}
