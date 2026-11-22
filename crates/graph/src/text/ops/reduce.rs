//! Text for reductions: `reduce_sum<axes=[1], keep_dims=keep>` and
//! `arg_max<axis=1, keep_dims=drop, tie=first>`.

use super::OpText;
use crate::ops::{ArgKind, ArgReduceAttrs, KeepDims, Op, ReduceAttrs, ReduceOp, TieBreak};
use crate::text::attr::{Attr, AttrValue, Attrs, TextWord, text_words};
use crate::text::error::ParseError;

text_words! {
    ReduceOp {
        ReduceOp::Sum => "reduce_sum",
        ReduceOp::Mean => "reduce_mean",
        ReduceOp::Prod => "reduce_prod",
        ReduceOp::Max => "reduce_max",
        ReduceOp::Min => "reduce_min",
        ReduceOp::SumSquare => "reduce_sum_square",
        ReduceOp::L1 => "reduce_l1",
        ReduceOp::L2 => "reduce_l2",
        ReduceOp::LogSum => "reduce_log_sum",
        ReduceOp::LogSumExp => "reduce_log_sum_exp",
    }
    ArgKind {
        ArgKind::Max => "arg_max",
        ArgKind::Min => "arg_min",
    }
    KeepDims {
        KeepDims::Keep => "keep",
        KeepDims::Drop => "drop",
    }
    TieBreak {
        TieBreak::First => "first",
        TieBreak::Last => "last",
    }
}

pub(super) fn encode_reduce(attrs: &ReduceAttrs) -> OpText {
    OpText::new(
        attrs.op.word(),
        vec![
            ("axes", AttrValue::list(attrs.axes.iter(), |axis| AttrValue::int(axis.index()))),
            ("keep_dims", AttrValue::word(attrs.keep_dims.word())),
        ],
    )
}

pub(super) fn encode_arg_reduce(attrs: ArgReduceAttrs) -> OpText {
    OpText::new(
        attrs.kind.word(),
        vec![
            ("axis", AttrValue::int(attrs.axis.index())),
            ("keep_dims", AttrValue::word(attrs.keep_dims.word())),
            ("tie", AttrValue::word(attrs.tie.word())),
        ],
    )
}

pub(super) fn decode(name: &str, attrs: &mut Attrs) -> Option<Result<Op, ParseError>> {
    if let Some(op) = ReduceOp::from_word(name) {
        return Some(decode_reduce(op, attrs));
    }
    ArgKind::from_word(name).map(|kind| decode_arg_reduce(kind, attrs))
}

fn decode_reduce(op: ReduceOp, attrs: &mut Attrs) -> Result<Op, ParseError> {
    let axes = attrs.take("axes")?.items()?.iter().map(Attr::axis).collect::<Result<_, _>>()?;
    let keep_dims = attrs.take("keep_dims")?.enumerated()?;
    Ok(Op::Reduce(ReduceAttrs { op, axes, keep_dims }))
}

fn decode_arg_reduce(kind: ArgKind, attrs: &mut Attrs) -> Result<Op, ParseError> {
    Ok(Op::ArgReduce(ArgReduceAttrs {
        kind,
        axis: attrs.take("axis")?.axis()?,
        keep_dims: attrs.take("keep_dims")?.enumerated()?,
        tie: attrs.take("tie")?.enumerated()?,
    }))
}
