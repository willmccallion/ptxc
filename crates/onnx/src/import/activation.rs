//! Imports activation functions, `Clip` and `PRelu`.

use crate::context::NodeContext;
use crate::error::ImportError;
use crate::lowered::{Lowered, Operand};
use ptxc_core::float::{Bf16, F16};
use ptxc_core::{DType, HostTensor, Shape, TensorData};
use ptxc_graph::{Activation, F32Attr, GeluApproximation, Op};

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    let activation = match op {
        "Relu" => Ok(Activation::Relu),
        "Sigmoid" => Ok(Activation::Sigmoid),
        "Tanh" => Ok(Activation::Tanh),
        "Softplus" => Ok(Activation::Softplus),
        "Softsign" => Ok(Activation::Softsign),
        "HardSwish" => Ok(Activation::HardSwish),
        "Mish" => Ok(Activation::Mish),
        "Gelu" => gelu(context),
        "LeakyRelu" => alpha(context, 0.01).map(|alpha| Activation::LeakyRelu { alpha }),
        "Elu" => alpha(context, 1.0).map(|alpha| Activation::Elu { alpha }),
        "Celu" => alpha(context, 1.0).map(|alpha| Activation::Celu { alpha }),
        "ThresholdedRelu" => alpha(context, 1.0).map(|alpha| Activation::ThresholdedRelu { alpha }),
        "Selu" => selu(context),
        "HardSigmoid" => hard_sigmoid(context),
        "Clip" => return Some(clip(context)),
        "PRelu" => return Some(Ok(Lowered::on_inputs(Op::PRelu, 2))),
        _ => return None,
    };
    Some(activation.map(|activation| Lowered::on_inputs(Op::Activation(activation), 1)))
}

fn alpha(context: &mut NodeContext<'_>, default: f32) -> Result<F32Attr, ImportError> {
    context.attrs.float_or("alpha", default).map(F32Attr::new)
}

fn gelu(context: &mut NodeContext<'_>) -> Result<Activation, ImportError> {
    match context.attrs.string_or("approximate", "none")?.as_str() {
        "none" => Ok(Activation::Gelu(GeluApproximation::None)),
        "tanh" => Ok(Activation::Gelu(GeluApproximation::Tanh)),
        _ => Err(context.attrs.unsupported("approximate")),
    }
}

/// ONNX's defaults: float32 approximations of the SELU constants.
fn selu(context: &mut NodeContext<'_>) -> Result<Activation, ImportError> {
    let alpha = alpha(context, 1.673_263_2)?;
    let gamma = context.attrs.float_or("gamma", 1.050_701)?;
    Ok(Activation::Selu { alpha, gamma: F32Attr::new(gamma) })
}

fn hard_sigmoid(context: &mut NodeContext<'_>) -> Result<Activation, ImportError> {
    let alpha = alpha(context, 0.2)?;
    let beta = context.attrs.float_or("beta", 0.5)?;
    Ok(Activation::HardSigmoid { alpha, beta: F32Attr::new(beta) })
}

/// `Clip(x, min?, max?)`, with a missing bound replaced by the element
/// type's lowest or highest value so the graph always has both (D32).
fn clip(context: &NodeContext<'_>) -> Result<Lowered, ImportError> {
    let dtype = context.ty(0)?.dtype();
    let mut operands = vec![Operand::Input(0)];
    for (index, bound) in [(1, Bound::Lowest), (2, Bound::Highest)] {
        operands.push(if context.has_input(index) {
            Operand::Input(index)
        } else {
            Operand::Constant(extreme(dtype, bound, context)?)
        });
    }
    Ok(Lowered::Node { op: Op::Clip, operands })
}

#[derive(Clone, Copy)]
enum Bound {
    Lowest,
    Highest,
}

/// A scalar of `dtype` at its lowest or highest value; floats use infinity.
fn extreme(
    dtype: DType,
    bound: Bound,
    context: &NodeContext<'_>,
) -> Result<HostTensor, ImportError> {
    let low = matches!(bound, Bound::Lowest);
    let infinity = if low { f32::NEG_INFINITY } else { f32::INFINITY };
    let data: TensorData = match dtype {
        DType::F32 => vec![infinity].into(),
        DType::F16 => vec![F16::from_f32(infinity)].into(),
        DType::Bf16 => vec![Bf16::from_f32(infinity)].into(),
        DType::I8 => vec![if low { i8::MIN } else { i8::MAX }].into(),
        DType::U8 => vec![if low { u8::MIN } else { u8::MAX }].into(),
        DType::I32 => vec![if low { i32::MIN } else { i32::MAX }].into(),
        DType::I64 => vec![if low { i64::MIN } else { i64::MAX }].into(),
        DType::F8E4M3Fn | DType::F8E5M2 | DType::Bool => {
            return Err(crate::error::Unsupported::DType(format!("{dtype:?} for Clip")).into());
        }
    };
    HostTensor::new(Shape::scalar(), data).map_err(|_| context.attrs.unsupported("bound"))
}
