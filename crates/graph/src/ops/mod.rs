//! The `Op` enum and the `TypeRule` trait every op implements.
//!
//! Type rules define what ptxc's graph IR accepts, independent of opset: the
//! importer maps each ONNX opset's version of an op onto these.

mod accepts;
mod activation;
mod attention;
mod attr;
mod conv;
mod elementwise;
mod gather;
mod matmul;
mod norm;
mod pool;
mod quant;
mod reduce;
mod shape;
mod type_error;
mod window;

pub use activation::{Activation, GeluApproximation};
pub use attr::F32Attr;
pub use conv::{ConvAttrs, Groups};
pub use elementwise::{BinaryOp, CastAttrs, InfDetection, ModKind, UnaryOp, VariadicOp};
pub use matmul::{GemmAttrs, Orientation};
pub use norm::{
    BatchNormAttrs, InstanceNormAttrs, LayerNormAttrs, LayerNormKind, NormOutputs, SoftmaxAttrs,
    SoftmaxKind, StashType,
};
pub use pool::{GlobalPoolKind, PadCounting, PoolAttrs, PoolAxis, PoolKind};
pub use reduce::{ArgKind, ArgReduceAttrs, KeepDims, ReduceAttrs, ReduceOp, TieBreak};
pub use type_error::{OperandCount, TypeError};
pub use window::{Dilation, Padding, Rounding, Stride, WindowAxis};

use crate::tensor_type::TensorType;
use ptxc_core::{Axis, DType, Shape};

/// An operation on tensors. Attribute payloads are `Eq + Hash` (D2), so
/// structurally equal nodes can be found by hashing.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Op {
    /// An elementwise op on one tensor.
    Unary(UnaryOp),
    /// An elementwise op on two tensors, with ONNX's multidirectional
    /// broadcasting.
    Binary(BinaryOp),
    /// An elementwise op on one or more tensors, broadcast together.
    Variadic(VariadicOp),
    /// `Where(condition, x, y)`, broadcast together.
    Where,
    /// Conversion to another element type.
    Cast(CastAttrs),
    /// An activation on one tensor.
    Activation(Activation),
    /// `Clip(x, min, max)` with scalar bounds.
    Clip,
    /// `PRelu(x, slope)`, the slope broadcasting to `x`.
    PRelu,
    /// Batched matrix multiplication, as numpy's `matmul`.
    MatMul,
    /// `alpha * A' * B' + beta * C` on matrices.
    Gemm(GemmAttrs),
    /// Grouped N-d convolution with an optional bias.
    Conv(ConvAttrs),
    /// Max or average pooling over sliding windows.
    Pool(PoolAttrs),
    /// Max or average over all spatial axes.
    GlobalPool(GlobalPoolKind),
    /// A reduction over explicit axes.
    Reduce(ReduceAttrs),
    /// `ArgMax` or `ArgMin` along one axis.
    ArgReduce(ArgReduceAttrs),
    /// `Softmax` or `LogSoftmax` along one axis.
    Softmax(SoftmaxAttrs),
    /// Layer or RMS normalisation over the trailing axes.
    LayerNorm(LayerNormAttrs),
    /// Inference-mode batch normalisation.
    BatchNorm(BatchNormAttrs),
    /// Instance normalisation.
    InstanceNorm(InstanceNormAttrs),
}

/// Infers an op's output types from its operand types.
pub trait TypeRule {
    /// The output types, or why the operands are not accepted.
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError>;
}

impl TypeRule for Op {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        match self {
            Self::Unary(op) => op.infer(operands),
            Self::Binary(op) => op.infer(operands),
            Self::Variadic(op) => op.infer(operands),
            Self::Where => elementwise::infer_where(operands),
            Self::Cast(attrs) => attrs.infer(operands),
            Self::Activation(op) => op.infer(operands),
            Self::Clip => activation::infer_clip(operands),
            Self::PRelu => activation::infer_prelu(operands),
            Self::MatMul => matmul::infer_matmul(operands),
            Self::Gemm(attrs) => attrs.infer(operands),
            Self::Conv(attrs) => attrs.infer(operands),
            Self::Pool(attrs) => attrs.infer(operands),
            Self::GlobalPool(kind) => kind.infer(operands),
            Self::Reduce(attrs) => attrs.infer(operands),
            Self::ArgReduce(attrs) => attrs.infer(operands),
            Self::Softmax(attrs) => attrs.infer(operands),
            Self::LayerNorm(attrs) => attrs.infer(operands),
            Self::BatchNorm(attrs) => attrs.infer(operands),
            Self::InstanceNorm(attrs) => attrs.infer(operands),
        }
    }
}

/// The operands as `N` required ones and an optional last one.
fn operands_with_optional<'a, const N: usize>(
    operands: &[&'a TensorType],
) -> Result<([&'a TensorType; N], Option<&'a TensorType>), TypeError> {
    let arity_error = || TypeError::ArityBetween {
        min: OperandCount(N),
        max: OperandCount(N + 1),
        actual: OperandCount(operands.len()),
    };
    match operands.split_at_checked(N) {
        Some((required, [])) => Ok((required.try_into().map_err(|_| arity_error())?, None)),
        Some((required, [optional])) => {
            Ok((required.try_into().map_err(|_| arity_error())?, Some(optional)))
        }
        _ => Err(arity_error()),
    }
}

/// Checks that `axis` is one of `shape`'s axes.
const fn check_axis(shape: &Shape, axis: Axis) -> Result<(), TypeError> {
    if axis.index() < shape.rank().get() {
        Ok(())
    } else {
        Err(TypeError::AxisOutOfRange { axis, rank: shape.rank() })
    }
}

/// Checks that `operand` has the element type `expected`.
fn require_dtype(expected: DType, operand: &TensorType) -> Result<(), TypeError> {
    if operand.dtype() == expected {
        Ok(())
    } else {
        Err(TypeError::MixedDTypes { lhs: expected, rhs: operand.dtype() })
    }
}

/// The operands as an array of exactly `N`.
fn exact_operands<'a, const N: usize>(
    operands: &[&'a TensorType],
) -> Result<[&'a TensorType; N], TypeError> {
    operands.try_into().map_err(|_| TypeError::Arity {
        expected: OperandCount(N),
        actual: OperandCount(operands.len()),
    })
}
