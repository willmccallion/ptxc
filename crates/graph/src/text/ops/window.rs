//! Text for convolution and pooling: each spatial axis is a list,
//! `[stride, dilation, pad_begin, pad_end]`, with the kernel first for pools.

use super::OpText;
use crate::ops::{
    ConvAttrs, Dilation, GlobalPoolKind, Groups, Op, PadCounting, Padding, PoolAttrs, PoolAxis,
    PoolKind, Rounding, Stride, WindowAxis,
};
use crate::text::attr::{Attr, AttrValue, Attrs, TextWord, text_words};
use crate::text::error::ParseError;

text_words! {
    Rounding {
        Rounding::Floor => "floor",
        Rounding::Ceil => "ceil",
    }
    PadCounting {
        PadCounting::Exclude => "exclude",
        PadCounting::Include => "include",
    }
}

fn window_items(axis: WindowAxis) -> [AttrValue; 4] {
    [
        AttrValue::int(axis.stride.get()),
        AttrValue::int(axis.dilation.get()),
        AttrValue::int(axis.pad_begin.get()),
        AttrValue::int(axis.pad_end.get()),
    ]
}

/// `conv<groups=1, axes=[[1, 1, 0, 0], ...]>`, shared with `qconv`.
pub(super) fn conv_attrs(attrs: &ConvAttrs) -> Vec<(&'static str, AttrValue)> {
    vec![
        ("groups", AttrValue::int(attrs.groups.get())),
        (
            "axes",
            AttrValue::list(attrs.axes.iter(), |axis| {
                AttrValue::List(window_items(*axis).to_vec())
            }),
        ),
    ]
}

pub(super) fn encode_conv(attrs: &ConvAttrs) -> OpText {
    OpText::new("conv", conv_attrs(attrs))
}

pub(super) fn encode_pool(attrs: &PoolAttrs) -> OpText {
    let axes = AttrValue::list(attrs.axes.iter(), |axis| {
        let mut items = vec![AttrValue::int(axis.kernel.get())];
        items.extend(window_items(axis.window));
        AttrValue::List(items)
    });
    let rounding = ("rounding", AttrValue::word(attrs.rounding.word()));
    match attrs.kind {
        PoolKind::Max => OpText::new("max_pool", vec![rounding, ("axes", axes)]),
        PoolKind::Average(counting) => OpText::new(
            "average_pool",
            vec![("pads", AttrValue::word(counting.word())), rounding, ("axes", axes)],
        ),
    }
}

pub(super) const fn encode_global_pool(kind: GlobalPoolKind) -> OpText {
    match kind {
        GlobalPoolKind::Max => OpText::plain("global_max_pool"),
        GlobalPoolKind::Average => OpText::plain("global_average_pool"),
    }
}

pub(super) fn decode(name: &str, attrs: &mut Attrs) -> Option<Result<Op, ParseError>> {
    Some(match name {
        "conv" => decode_conv(attrs).map(Op::Conv),
        "max_pool" => decode_pool(attrs, PoolKind::Max),
        "average_pool" => attrs
            .take("pads")
            .and_then(|pads| pads.enumerated())
            .and_then(|c| decode_pool(attrs, PoolKind::Average(c))),
        "global_max_pool" => Ok(Op::GlobalPool(GlobalPoolKind::Max)),
        "global_average_pool" => Ok(Op::GlobalPool(GlobalPoolKind::Average)),
        _ => return None,
    })
}

/// Reads `groups` and `axes`, shared with `qconv`.
pub(super) fn decode_conv(attrs: &mut Attrs) -> Result<ConvAttrs, ParseError> {
    let groups = attrs.take("groups")?;
    let groups = Groups::new(groups.u64()?).ok_or_else(|| groups.bad("must be positive"))?;
    let axes = attrs
        .take("axes")?
        .items()?
        .iter()
        .map(|axis| decode_window(&axis.tuple::<4>()?))
        .collect::<Result<_, _>>()?;
    Ok(ConvAttrs { axes, groups })
}

fn decode_pool(attrs: &mut Attrs, kind: PoolKind) -> Result<Op, ParseError> {
    let rounding = attrs.take("rounding")?.enumerated()?;
    let axes = attrs
        .take("axes")?
        .items()?
        .iter()
        .map(|axis| {
            let [kernel, stride, dilation, begin, end] = axis.tuple::<5>()?;
            Ok(PoolAxis {
                kernel: kernel.dim()?,
                window: decode_window(&[stride, dilation, begin, end])?,
            })
        })
        .collect::<Result<_, ParseError>>()?;
    Ok(Op::Pool(PoolAttrs { kind, axes, rounding }))
}

fn decode_window([stride, dilation, begin, end]: &[Attr; 4]) -> Result<WindowAxis, ParseError> {
    Ok(WindowAxis {
        stride: Stride::new(stride.u64()?)
            .ok_or_else(|| stride.bad("a stride must be positive"))?,
        dilation: Dilation::new(dilation.u64()?)
            .ok_or_else(|| dilation.bad("a dilation must be positive"))?,
        pad_begin: Padding::new(begin.u64()?),
        pad_end: Padding::new(end.u64()?),
    })
}
