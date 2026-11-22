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

pub use activation::{Activation, GeluApproximation};
pub use attr::F32Attr;
pub use elementwise::{BinaryOp, CastAttrs, InfDetection, ModKind, UnaryOp, VariadicOp};
pub use matmul::{GemmAttrs, Orientation};

use crate::tensor_type::TensorType;
use ptxc_core::{DType, Dim, Rank, Shape, ShapeError};
use thiserror::Error;

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
        }
    }
}

/// A number of operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OperandCount(usize);

impl OperandCount {
    /// The count.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Operands an op's type rule rejects.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TypeError {
    /// The wrong number of operands.
    #[error("expected {} operands, got {}", .expected.get(), .actual.get())]
    Arity {
        /// What the op takes.
        expected: OperandCount,
        /// What it was given.
        actual: OperandCount,
    },
    /// A number of operands outside the op's range.
    #[error("expected {} to {} operands, got {}", .min.get(), .max.get(), .actual.get())]
    ArityBetween {
        /// The fewest the op takes.
        min: OperandCount,
        /// The most the op takes.
        max: OperandCount,
        /// What it was given.
        actual: OperandCount,
    },
    /// An operand of the wrong rank.
    #[error("expected rank {}, got rank {}", .expected.get(), .actual.get())]
    Rank {
        /// The rank the op needs.
        expected: Rank,
        /// The operand's rank.
        actual: Rank,
    },
    /// An operand of too low a rank.
    #[error("expected rank at least {}, got rank {}", .minimum.get(), .actual.get())]
    RankBelow {
        /// The lowest rank the op takes.
        minimum: Rank,
        /// The operand's rank.
        actual: Rank,
    },
    /// The dims a matrix product contracts differ.
    #[error("contracted dims differ: {} and {}", .lhs.get(), .rhs.get())]
    ContractedDims {
        /// The left operand's contracted dim.
        lhs: Dim,
        /// The right operand's contracted dim.
        rhs: Dim,
    },
    /// An op that needs at least one operand was given none.
    #[error("expected at least one operand")]
    NoOperands,
    /// Operands that must share an element type do not.
    #[error("operands have different element types: {lhs:?} and {rhs:?}")]
    MixedDTypes {
        /// The first operand's element type.
        lhs: DType,
        /// The second operand's element type.
        rhs: DType,
    },
    /// An element type the op does not accept.
    #[error("element type {0:?} is not accepted")]
    UnsupportedDType(DType),
    /// Shapes that do not broadcast together.
    #[error("shapes {:?} and {:?} do not broadcast", .lhs.dims(), .rhs.dims())]
    NotBroadcastable {
        /// The first operand's shape.
        lhs: Shape,
        /// The second operand's shape.
        rhs: Shape,
    },
    /// An operand that must be a scalar has a shape.
    #[error("expected a scalar, got shape {:?}", .0.dims())]
    NotScalar(Shape),
    /// A shape that must broadcast to another, without changing it, does not.
    #[error("shape {:?} does not broadcast to {:?}", .from.dims(), .to.dims())]
    NotUnidirectional {
        /// The shape being broadcast.
        from: Shape,
        /// The shape it must broadcast to.
        to: Shape,
    },
    /// An output shape that cannot exist.
    #[error(transparent)]
    Shape(#[from] ShapeError),
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
