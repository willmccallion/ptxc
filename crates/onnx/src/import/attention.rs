//! Imports ONNX `Attention` (opset 23+).

use crate::context::NodeContext;
use crate::error::{ImportError, ImportErrorKind};
use crate::lowered::{Lowered, Operand};
use crate::tensor::dtype_from_onnx;
use ptxc_core::DType;
use ptxc_graph::{
    AttentionAttrs, AttentionLayout, AttentionMask, Causality, F32Attr, HeadCount, Op, Presence,
    QkOutput,
};

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    (op == "Attention").then(|| attention(context))
}

fn invalid(problem: &str) -> ImportError {
    ImportErrorKind::Invalid { field: "Attention", problem: problem.to_owned() }.into()
}

const fn presence(present: bool) -> Presence {
    if present { Presence::Present } else { Presence::Absent }
}

/// `Attention(Q, K, V, attn_mask?, past_key?, past_value?, nonpad_kv_seqlen?)`.
/// Q's rank picks the layout; the default scale is `1 / sqrt(head_size)` and
/// the default softmax precision is Q's type. The present key and value are
/// produced whenever a later output is wanted, so outputs keep ONNX's
/// positions.
fn attention(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let q = context.ty(0)?.clone();
    let causal = context.attrs.int_or("is_causal", 0)? != 0;
    let q_heads = context.attrs.int("q_num_heads")?;
    let kv_heads = context.attrs.int("kv_num_heads")?;
    let qk_mode = context.attrs.int_or("qk_matmul_output_mode", 0)?;
    let scale = context.attrs.float("scale")?;
    let softcap = context.attrs.float_or("softcap", 0.0)?;
    let precision = context.attrs.int("softmax_precision")?;
    let (layout, head_size) = layout(&q, q_heads, kv_heads)?;
    let past = past(context)?;
    let mask = match context.has_input(3).then(|| context.ty(3)).transpose()? {
        None => AttentionMask::None,
        Some(ty) if ty.dtype() == DType::Bool => AttentionMask::Boolean,
        Some(_) => AttentionMask::Additive,
    };
    let qk_output = context
        .wants_output(3)
        .then(|| qk_output(qk_mode))
        .transpose()
        .map_err(|()| context.attrs.unsupported("qk_matmul_output_mode"))?;
    let attrs = AttentionAttrs {
        layout,
        mask,
        causality: if causal { Causality::Causal } else { Causality::Full },
        past: presence(past),
        present: presence((1..=3).any(|index| context.wants_output(index))),
        nonpad_lengths: presence(context.has_input(6)),
        qk_output,
        scale: F32Attr::new(scale.map_or_else(|| default_scale(head_size), Ok)?),
        softcap: (softcap != 0.0).then_some(F32Attr::new(softcap)),
        softmax_precision: match precision {
            Some(code) => {
                dtype_from_onnx(i32::try_from(code).map_err(|_| invalid("softmax_precision"))?)?
            }
            None => q.dtype(),
        },
    };
    let mut operands = vec![Operand::Input(0), Operand::Input(1), Operand::Input(2)];
    operands.extend(
        [3, 4, 5, 6].into_iter().filter(|&index| context.has_input(index)).map(Operand::Input),
    );
    Ok(Lowered::Node { op: Op::Attention(attrs), operands })
}

/// The layout Q's rank gives, and the head size.
fn layout(
    q: &ptxc_graph::TensorType,
    q_heads: Option<i64>,
    kv_heads: Option<i64>,
) -> Result<(AttentionLayout, u64), ImportError> {
    let heads = |count: Option<i64>| {
        count
            .and_then(|c| u64::try_from(c).ok())
            .and_then(HeadCount::new)
            .ok_or_else(|| invalid("3-D inputs need q_num_heads and kv_num_heads"))
    };
    match q.shape().dims() {
        [_, _, _, head] => Ok((AttentionLayout::Heads, head.get())),
        [_, _, hidden] => {
            let q_heads = heads(q_heads)?;
            Ok((
                AttentionLayout::Packed { q_heads, kv_heads: heads(kv_heads)? },
                hidden.get() / q_heads.get(),
            ))
        }
        _ => Err(invalid("Q must be 3-D or 4-D")),
    }
}

/// Whether the past key and value are given; one without the other is not
/// supported.
fn past(context: &NodeContext<'_>) -> Result<bool, ImportError> {
    match (context.has_input(4), context.has_input(5)) {
        (true, true) => Ok(true),
        (false, false) => Ok(false),
        _ => Err(context.attrs.unsupported("past_key without past_value")),
    }
}

const fn qk_output(mode: i64) -> Result<QkOutput, ()> {
    match mode {
        0 => Ok(QkOutput::MatMul),
        1 => Ok(QkOutput::Softcapped),
        2 => Ok(QkOutput::Masked),
        3 => Ok(QkOutput::Softmax),
        _ => Err(()),
    }
}

/// `1 / sqrt(head_size)`, ONNX's default scale.
fn default_scale(head_size: u64) -> Result<f32, ImportError> {
    let size =
        u16::try_from(head_size).map(f32::from).map_err(|_| invalid("head size too large"))?;
    Ok(1.0 / size.sqrt())
}
