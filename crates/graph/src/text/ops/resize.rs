//! Text for `resize`. Each mode's own attributes appear only with it:
//! `nearest` with nearest, `cubic_a` and `exclude_outside` with cubic,
//! `roi` and `extrapolation` with `tf_crop_and_resize`.

use super::OpText;
use crate::ops::{
    Antialias, CoordinateTransform, NearestRounding, Op, OutsideWeights, ResizeAttrs, ResizeAxis,
    ResizeMode,
};
use crate::text::attr::{AttrValue, Attrs, TextWord, text_words};
use crate::text::error::ParseError;

text_words! {
    NearestRounding {
        NearestRounding::RoundPreferFloor => "round_prefer_floor",
        NearestRounding::RoundPreferCeil => "round_prefer_ceil",
        NearestRounding::Floor => "floor",
        NearestRounding::Ceil => "ceil",
    }
    OutsideWeights {
        OutsideWeights::Include => "include",
        OutsideWeights::Exclude => "exclude",
    }
    Antialias {
        Antialias::Off => "off",
        Antialias::On => "on",
    }
}

pub(super) fn encode(attrs: &ResizeAttrs) -> OpText {
    let mut fields = Vec::new();
    match attrs.mode {
        ResizeMode::Nearest(rounding) => {
            fields.push(("mode", AttrValue::word("nearest")));
            fields.push(("nearest", AttrValue::word(rounding.word())));
        }
        ResizeMode::Linear => fields.push(("mode", AttrValue::word("linear"))),
        ResizeMode::Cubic { coefficient, outside } => {
            fields.push(("mode", AttrValue::word("cubic")));
            fields.push(("cubic_a", AttrValue::float(coefficient.get())));
            fields.push(("exclude_outside", AttrValue::word(outside.word())));
        }
    }
    encode_coordinates(&attrs.coordinates, &mut fields);
    fields.push(("antialias", AttrValue::word(attrs.antialias.word())));
    let axes = AttrValue::list(attrs.axes.iter(), |axis| {
        AttrValue::List(vec![AttrValue::int(axis.size.get()), AttrValue::float(axis.scale.get())])
    });
    fields.push(("axes", axes));
    OpText::new("resize", fields)
}

fn encode_coordinates(
    coordinates: &CoordinateTransform,
    fields: &mut Vec<(&'static str, AttrValue)>,
) {
    let word = match coordinates {
        CoordinateTransform::HalfPixel => "half_pixel",
        CoordinateTransform::HalfPixelSymmetric => "half_pixel_symmetric",
        CoordinateTransform::PytorchHalfPixel => "pytorch_half_pixel",
        CoordinateTransform::AlignCorners => "align_corners",
        CoordinateTransform::Asymmetric => "asymmetric",
        CoordinateTransform::TfCropAndResize { roi, extrapolation } => {
            fields.push(("coordinates", AttrValue::word("tf_crop_and_resize")));
            let regions = AttrValue::list(roi.iter(), |[start, end]| {
                AttrValue::List(vec![AttrValue::float(start.get()), AttrValue::float(end.get())])
            });
            fields.push(("roi", regions));
            fields.push(("extrapolation", AttrValue::float(extrapolation.get())));
            return;
        }
    };
    fields.push(("coordinates", AttrValue::word(word)));
}

pub(super) fn decode(name: &str, attrs: &mut Attrs) -> Option<Result<Op, ParseError>> {
    (name == "resize").then(|| decode_resize(attrs))
}

fn decode_resize(attrs: &mut Attrs) -> Result<Op, ParseError> {
    let mode_attr = attrs.take("mode")?;
    let mode = match mode_attr.word()? {
        "nearest" => ResizeMode::Nearest(attrs.take("nearest")?.enumerated()?),
        "linear" => ResizeMode::Linear,
        "cubic" => ResizeMode::Cubic {
            coefficient: attrs.take("cubic_a")?.f32_attr()?,
            outside: attrs.take("exclude_outside")?.enumerated()?,
        },
        _ => return Err(mode_attr.bad("expected `nearest`, `linear` or `cubic`")),
    };
    let coordinates = decode_coordinates(attrs)?;
    let antialias = attrs.take("antialias")?.enumerated()?;
    let axes = attrs
        .take("axes")?
        .items()?
        .iter()
        .map(|axis| {
            let [size, scale] = axis.tuple::<2>()?;
            Ok(ResizeAxis { size: size.dim()?, scale: scale.f32_attr()? })
        })
        .collect::<Result<_, ParseError>>()?;
    Ok(Op::Resize(ResizeAttrs { mode, coordinates, antialias, axes }))
}

fn decode_coordinates(attrs: &mut Attrs) -> Result<CoordinateTransform, ParseError> {
    let attr = attrs.take("coordinates")?;
    Ok(match attr.word()? {
        "half_pixel" => CoordinateTransform::HalfPixel,
        "half_pixel_symmetric" => CoordinateTransform::HalfPixelSymmetric,
        "pytorch_half_pixel" => CoordinateTransform::PytorchHalfPixel,
        "align_corners" => CoordinateTransform::AlignCorners,
        "asymmetric" => CoordinateTransform::Asymmetric,
        "tf_crop_and_resize" => {
            let roi = attrs
                .take("roi")?
                .items()?
                .iter()
                .map(|region| {
                    let [start, end] = region.tuple::<2>()?;
                    Ok([start.f32_attr()?, end.f32_attr()?])
                })
                .collect::<Result<_, ParseError>>()?;
            let extrapolation = attrs.take("extrapolation")?.f32_attr()?;
            CoordinateTransform::TfCropAndResize { roi, extrapolation }
        }
        _ => return Err(attr.bad("not a coordinate transformation mode")),
    })
}
