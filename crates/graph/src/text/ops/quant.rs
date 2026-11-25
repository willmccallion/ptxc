//! Text for quantization ops. Parameters are `[u8, [scale, zero_point]]`
//! per tensor, or `[i8, axis, [[scale, zero_point], ...]]` per axis.

use super::OpText;
use super::window::{conv_attrs, decode_conv};
use crate::ops::{DequantizeAttrs, Op, QConvAttrs, QMatMulAttrs, QMatMulOutput, QuantizeAttrs};
use crate::text::attr::{Attr, AttrValue, Attrs, TextWord, text_words};
use crate::text::error::ParseError;
use ptxc_core::float::Overflow;
use ptxc_core::{Affine, Granularity, QuantDType, QuantParams, Scale, ZeroPoint};

text_words! {
    QuantDType {
        QuantDType::I8 => "i8",
        QuantDType::U8 => "u8",
        QuantDType::I32 => "i32",
        QuantDType::F8E4M3Fn => "f8e4m3fn",
        QuantDType::F8E5M2 => "f8e5m2",
    }
}

fn affine_value(map: Affine) -> AttrValue {
    AttrValue::List(vec![AttrValue::float(map.scale.get()), AttrValue::int(map.zero_point.get())])
}

fn params_value(params: &QuantParams) -> AttrValue {
    let dtype = AttrValue::word(params.dtype().word());
    match params.granularity() {
        Granularity::PerTensor(map) => AttrValue::List(vec![dtype, affine_value(*map)]),
        Granularity::PerAxis { axis, maps } => AttrValue::List(vec![
            dtype,
            AttrValue::int(axis.index()),
            AttrValue::list(maps.iter().copied(), affine_value),
        ]),
    }
}

pub(super) fn encode_quantize(attrs: &QuantizeAttrs) -> OpText {
    let mut fields = vec![("params", params_value(attrs.params()))];
    if matches!(attrs.params().dtype(), QuantDType::F8E4M3Fn | QuantDType::F8E5M2) {
        fields.push(("overflow", AttrValue::word(attrs.overflow().word())));
    }
    OpText::new("quantize", fields)
}

pub(super) fn encode_dequantize(attrs: &DequantizeAttrs) -> OpText {
    OpText::new(
        "dequantize",
        vec![
            ("params", params_value(&attrs.params)),
            ("output", AttrValue::word(attrs.output.word())),
        ],
    )
}

/// A quantized result is written as `output=` its grid; a float one as
/// `result=` its type.
pub(super) fn encode_qmatmul(attrs: &QMatMulAttrs) -> OpText {
    let output = match &attrs.output {
        QMatMulOutput::Quantized(params) => ("output", params_value(params)),
        QMatMulOutput::Float(dtype) => ("result", AttrValue::word(dtype.word())),
    };
    OpText::new(
        "qmatmul",
        vec![("a", params_value(&attrs.a)), ("b", params_value(&attrs.b)), output],
    )
}

pub(super) fn encode_qconv(attrs: &QConvAttrs) -> OpText {
    let mut fields = conv_attrs(&attrs.conv);
    fields.extend([
        ("input", params_value(&attrs.input)),
        ("weights", params_value(&attrs.weights)),
        ("output", params_value(&attrs.output)),
    ]);
    OpText::new("qconv", fields)
}

pub(super) fn decode(name: &str, attrs: &mut Attrs) -> Option<Result<Op, ParseError>> {
    Some(match name {
        "quantize" => decode_quantize(attrs),
        "dequantize" => decode_dequantize(attrs),
        "qmatmul" => decode_qmatmul(attrs),
        "qconv" => decode_qconv(attrs),
        _ => return None,
    })
}

/// `overflow` is written only for FP8 grids, where it matters.
fn decode_quantize(attrs: &mut Attrs) -> Result<Op, ParseError> {
    let params = decode_params(&attrs.take("params")?)?;
    let overflow = match attrs.take_optional("overflow") {
        Some(overflow) => overflow.enumerated()?,
        None => Overflow::Saturate,
    };
    Ok(Op::Quantize(QuantizeAttrs::new(params, overflow)))
}

fn decode_dequantize(attrs: &mut Attrs) -> Result<Op, ParseError> {
    let params = decode_params(&attrs.take("params")?)?;
    let output = attrs.take("output")?.enumerated()?;
    Ok(Op::Dequantize(DequantizeAttrs { params, output }))
}

fn decode_qmatmul(attrs: &mut Attrs) -> Result<Op, ParseError> {
    let (a, b) = (decode_params(&attrs.take("a")?)?, decode_params(&attrs.take("b")?)?);
    let output = match attrs.take_optional("output") {
        Some(params) => QMatMulOutput::Quantized(decode_params(&params)?),
        None => QMatMulOutput::Float(attrs.take("result")?.enumerated()?),
    };
    Ok(Op::QMatMul(QMatMulAttrs { a, b, output }))
}

fn decode_qconv(attrs: &mut Attrs) -> Result<Op, ParseError> {
    Ok(Op::QConv(QConvAttrs {
        conv: decode_conv(attrs)?,
        input: decode_params(&attrs.take("input")?)?,
        weights: decode_params(&attrs.take("weights")?)?,
        output: decode_params(&attrs.take("output")?)?,
    }))
}

fn decode_params(attr: &Attr) -> Result<QuantParams, ParseError> {
    let items = attr.items()?;
    let granularity = match items.as_slice() {
        [_, map] => Granularity::PerTensor(decode_affine(map)?),
        [_, axis, maps] => Granularity::PerAxis {
            axis: axis.axis()?,
            maps: maps.items()?.iter().map(decode_affine).collect::<Result<_, _>>()?,
        },
        _ => return Err(attr.bad("expected [dtype, map] or [dtype, axis, maps]")),
    };
    let dtype = items[0].enumerated()?;
    QuantParams::new(dtype, granularity).map_err(|_| attr.bad("a zero point is out of range"))
}

fn decode_affine(map: &Attr) -> Result<Affine, ParseError> {
    let [scale, zero_point] = map.tuple::<2>()?;
    let zero_point = i32::try_from(zero_point.i64()?)
        .map_err(|_| zero_point.bad("a zero point must fit in i32"))?;
    Ok(Affine { scale: Scale::new(scale.f32()?), zero_point: ZeroPoint::new(zero_point) })
}
