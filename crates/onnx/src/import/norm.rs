//! Imports softmax and batch, instance, layer and RMS normalisation.

use crate::context::NodeContext;
use crate::error::ImportError;
use crate::lowered::Lowered;
use ptxc_graph::{
    BatchNormAttrs, F32Attr, InstanceNormAttrs, LayerNormAttrs, LayerNormKind, NormOutputs, Op,
    SoftmaxAttrs, SoftmaxKind, StashType,
};
use ptxc_onnx_proto::tensor_proto::DataType;

/// ONNX's default epsilon for every normalisation.
const EPSILON: f32 = 1e-5;

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    Some(match op {
        "Softmax" => softmax(context, SoftmaxKind::Softmax),
        "LogSoftmax" => softmax(context, SoftmaxKind::LogSoftmax),
        "LayerNormalization" => layer_norm(context),
        "RMSNormalization" => rms_norm(context),
        "BatchNormalization" => batch_norm(context),
        "InstanceNormalization" => epsilon(context)
            .map(|epsilon| Lowered::on_inputs(Op::InstanceNorm(InstanceNormAttrs { epsilon }), 3)),
        _ => return None,
    })
}

fn epsilon(context: &mut NodeContext<'_>) -> Result<F32Attr, ImportError> {
    context.attrs.float_or("epsilon", EPSILON).map(F32Attr::new)
}

/// Opset 13 onwards: one axis, defaulting to the last.
fn softmax(context: &mut NodeContext<'_>, kind: SoftmaxKind) -> Result<Lowered, ImportError> {
    let rank = context.ty(0)?.shape().rank();
    let axis = context.axis_attr("axis", -1, rank)?;
    Ok(Lowered::on_inputs(Op::Softmax(SoftmaxAttrs { kind, axis }), 1))
}

/// The shared attributes of `LayerNormalization` and `RMSNormalization`.
fn layer_attrs(
    context: &mut NodeContext<'_>,
    kind: LayerNormKind,
) -> Result<LayerNormAttrs, ImportError> {
    let rank = context.ty(0)?.shape().rank();
    let axis = context.axis_attr("axis", -1, rank)?;
    let epsilon = epsilon(context)?;
    let stash = match context.attrs.int_or("stash_type", DataType::Float as i64)? {
        code if code == DataType::Float as i64 => StashType::F32,
        code if code == DataType::Bfloat16 as i64 => StashType::Bf16,
        _ => return Err(context.attrs.unsupported("stash_type")),
    };
    Ok(LayerNormAttrs { kind, axis, epsilon, stash })
}

/// `LayerNormalization`, with the mean and inverse standard deviation
/// outputs when the model asks for either.
fn layer_norm(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let outputs = if context.wants_output(1) || context.wants_output(2) {
        NormOutputs::WithStatistics
    } else {
        NormOutputs::Normalized
    };
    let attrs = layer_attrs(context, LayerNormKind::Layer(outputs))?;
    let operands = if context.has_input(2) { 3 } else { 2 };
    Ok(Lowered::on_inputs(Op::LayerNorm(attrs), operands))
}

fn rms_norm(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let attrs = layer_attrs(context, LayerNormKind::Rms)?;
    Ok(Lowered::on_inputs(Op::LayerNorm(attrs), 2))
}

/// Inference-mode `BatchNormalization`. `momentum` only affects training,
/// which is not supported.
fn batch_norm(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let epsilon = epsilon(context)?;
    let _momentum: Option<f32> = context.attrs.float("momentum")?;
    if context.attrs.int_or("training_mode", 0)? != 0 {
        return Err(context.attrs.unsupported("training_mode"));
    }
    Ok(Lowered::on_inputs(Op::BatchNorm(BatchNormAttrs { epsilon }), 5))
}
