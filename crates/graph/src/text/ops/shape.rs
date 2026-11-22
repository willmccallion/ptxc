//! Text for shape ops and single-axis ops: `reshape<shape=[6, 4]>`,
//! `slice<axes=[[axis, start, step, length], ...]>`, `concat<axis=1>`,
//! `gather<axis=0>`.

use super::OpText;
use crate::ops::{
    AxisPadding, DiagonalOffset, Op, PadAttrs, PadMode, Padding, RepeatCount, SliceAttrs,
    SliceAxis, SliceStep, SplitAttrs, TileAttrs, TransposeAttrs, TriangularPart, TriluAttrs,
};
use crate::text::attr::{Attr, AttrValue, Attrs, TextWord, text_words};
use crate::text::error::ParseError;
use ptxc_core::{Axis, Shape};

text_words! {
    PadMode {
        PadMode::Constant => "constant",
        PadMode::Reflect => "reflect",
        PadMode::Edge => "edge",
        PadMode::Wrap => "wrap",
    }
    TriangularPart {
        TriangularPart::Upper => "upper",
        TriangularPart::Lower => "lower",
    }
}

fn axes_value(axes: &[Axis]) -> AttrValue {
    AttrValue::list(axes.iter(), |axis| AttrValue::int(axis.index()))
}

pub(super) fn encode_target(name: &'static str, target: &Shape) -> OpText {
    OpText::new(
        name,
        vec![("shape", AttrValue::list(target.dims().iter(), |dim| AttrValue::int(dim.get())))],
    )
}

pub(super) fn encode_axis(name: &'static str, axis: Axis) -> OpText {
    OpText::new(name, vec![("axis", AttrValue::int(axis.index()))])
}

pub(super) fn encode_transpose(attrs: &TransposeAttrs) -> OpText {
    OpText::new("transpose", vec![("permutation", axes_value(&attrs.permutation))])
}

pub(super) fn encode_tile(attrs: &TileAttrs) -> OpText {
    OpText::new(
        "tile",
        vec![(
            "repeats",
            AttrValue::list(attrs.repeats.iter(), |repeat| AttrValue::int(repeat.get())),
        )],
    )
}

pub(super) fn encode_slice(attrs: &SliceAttrs) -> OpText {
    let axes = AttrValue::list(attrs.axes.iter(), |slice| {
        AttrValue::List(vec![
            AttrValue::int(slice.axis.index()),
            AttrValue::int(slice.start),
            AttrValue::int(slice.step.get()),
            AttrValue::int(slice.length.get()),
        ])
    });
    OpText::new("slice", vec![("axes", axes)])
}

pub(super) fn encode_split(attrs: &SplitAttrs) -> OpText {
    OpText::new(
        "split",
        vec![
            ("axis", AttrValue::int(attrs.axis.index())),
            ("sizes", AttrValue::list(attrs.sizes.iter(), |size| AttrValue::int(size.get()))),
        ],
    )
}

pub(super) fn encode_pad(attrs: &PadAttrs) -> OpText {
    let pads = AttrValue::list(attrs.axes.iter(), |padding| {
        AttrValue::List(vec![
            AttrValue::int(padding.begin.get()),
            AttrValue::int(padding.end.get()),
        ])
    });
    OpText::new("pad", vec![("mode", AttrValue::word(attrs.mode.word())), ("pads", pads)])
}

pub(super) fn encode_trilu(attrs: TriluAttrs) -> OpText {
    OpText::new(
        "trilu",
        vec![
            ("part", AttrValue::word(attrs.part.word())),
            ("diagonal", AttrValue::int(attrs.diagonal.get())),
        ],
    )
}

pub(super) fn decode(name: &str, attrs: &mut Attrs) -> Option<Result<Op, ParseError>> {
    Some(match name {
        "reshape" => decode_target(attrs).map(Op::Reshape),
        "expand" => decode_target(attrs).map(Op::Expand),
        "concat" => decode_axis(attrs).map(Op::Concat),
        "gather" => decode_axis(attrs).map(Op::Gather),
        "gather_elements" => decode_axis(attrs).map(Op::GatherElements),
        "transpose" => decode_axes(attrs, "permutation")
            .map(|permutation| Op::Transpose(TransposeAttrs { permutation })),
        "tile" => decode_tile(attrs),
        "slice" => decode_slice(attrs),
        "split" => decode_split(attrs),
        "pad" => decode_pad(attrs),
        "trilu" => decode_trilu(attrs),
        _ => return None,
    })
}

fn decode_target(attrs: &mut Attrs) -> Result<Shape, ParseError> {
    let shape = attrs.take("shape")?;
    let dims = shape.items()?.iter().map(Attr::dim).collect::<Result<_, _>>()?;
    Shape::new(dims).map_err(|_| shape.bad("has more than 2^64 elements"))
}

fn decode_axis(attrs: &mut Attrs) -> Result<Axis, ParseError> {
    attrs.take("axis")?.axis()
}

fn decode_axes(attrs: &mut Attrs, key: &'static str) -> Result<Vec<Axis>, ParseError> {
    attrs.take(key)?.items()?.iter().map(Attr::axis).collect()
}

fn decode_tile(attrs: &mut Attrs) -> Result<Op, ParseError> {
    let repeats = attrs
        .take("repeats")?
        .items()?
        .iter()
        .map(|repeat| repeat.u64().map(RepeatCount::new))
        .collect::<Result<_, _>>()?;
    Ok(Op::Tile(TileAttrs { repeats }))
}

fn decode_slice(attrs: &mut Attrs) -> Result<Op, ParseError> {
    let axes = attrs
        .take("axes")?
        .items()?
        .iter()
        .map(|slice| {
            let [axis, start, step, length] = slice.tuple::<4>()?;
            Ok(SliceAxis {
                axis: axis.axis()?,
                start: start.u64()?,
                step: SliceStep::new(step.i64()?)
                    .ok_or_else(|| step.bad("a step must not be zero"))?,
                length: length.dim()?,
            })
        })
        .collect::<Result<_, ParseError>>()?;
    Ok(Op::Slice(SliceAttrs { axes }))
}

fn decode_split(attrs: &mut Attrs) -> Result<Op, ParseError> {
    let axis = decode_axis(attrs)?;
    let sizes = attrs.take("sizes")?.items()?.iter().map(Attr::dim).collect::<Result<_, _>>()?;
    Ok(Op::Split(SplitAttrs { axis, sizes }))
}

fn decode_pad(attrs: &mut Attrs) -> Result<Op, ParseError> {
    let mode = attrs.take("mode")?.enumerated()?;
    let axes = attrs
        .take("pads")?
        .items()?
        .iter()
        .map(|pair| {
            let [begin, end] = pair.tuple::<2>()?;
            Ok(AxisPadding { begin: Padding::new(begin.u64()?), end: Padding::new(end.u64()?) })
        })
        .collect::<Result<_, ParseError>>()?;
    Ok(Op::Pad(PadAttrs { mode, axes }))
}

fn decode_trilu(attrs: &mut Attrs) -> Result<Op, ParseError> {
    Ok(Op::Trilu(TriluAttrs {
        part: attrs.take("part")?.enumerated()?,
        diagonal: DiagonalOffset::new(attrs.take("diagonal")?.i64()?),
    }))
}
