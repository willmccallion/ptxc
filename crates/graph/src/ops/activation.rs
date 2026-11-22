//! Activation ops and their attributes.

use super::accepts::Accepts;
use super::attr::F32Attr;
use super::elementwise::broadcast_to;
use super::{TypeError, TypeRule, exact_operands};
use crate::tensor_type::{Layout, TensorType};

/// An activation on one tensor, keeping its type. Attributes are ONNX's,
/// with ONNX's defaults filled in by the importer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Activation {
    /// `max(x, 0)`.
    Relu,
    /// `1 / (1 + e^-x)`.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// `ln(1 + e^x)`.
    Softplus,
    /// `x / (1 + |x|)`.
    Softsign,
    /// `x * max(0, min(1, x / 6 + 1 / 2))`.
    HardSwish,
    /// `x * tanh(softplus(x))`.
    Mish,
    /// The Gaussian error linear unit.
    Gelu(GeluApproximation),
    /// `x` if `x >= 0`, else `alpha * x`.
    LeakyRelu {
        /// The slope below zero.
        alpha: F32Attr,
    },
    /// `x` if `x >= 0`, else `alpha * (e^x - 1)`.
    Elu {
        /// The scale below zero.
        alpha: F32Attr,
    },
    /// `max(0, x) + min(0, alpha * (e^(x / alpha) - 1))`.
    Celu {
        /// The scale and divisor below zero.
        alpha: F32Attr,
    },
    /// `x` if `x > alpha`, else 0.
    ThresholdedRelu {
        /// The threshold.
        alpha: F32Attr,
    },
    /// `gamma * x` if `x > 0`, else `gamma * alpha * (e^x - 1)`.
    Selu {
        /// The scale below zero.
        alpha: F32Attr,
        /// The overall scale.
        gamma: F32Attr,
    },
    /// `max(0, min(1, alpha * x + beta))`.
    HardSigmoid {
        /// The slope.
        alpha: F32Attr,
        /// The offset.
        beta: F32Attr,
    },
}

/// How `Gelu` computes the Gaussian CDF, from ONNX's `approximate`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GeluApproximation {
    /// `approximate = "none"`: exactly, with `erf`.
    None,
    /// `approximate = "tanh"`: with the tanh approximation.
    Tanh,
}

impl Activation {
    const fn accepts(self) -> Accepts {
        match self {
            Self::Relu => Accepts::SignedNumeric,
            _ => Accepts::Float,
        }
    }
}

impl TypeRule for Activation {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        self.accepts().check(input.dtype())?;
        Ok(vec![TensorType::new(input.dtype(), input.shape().clone(), Layout::RowMajor)])
    }
}

/// `Clip(x, min, max)`: `x` limited to `[min, max]`. The bounds are scalars
/// of `x`'s element type; the importer supplies the type's lowest or highest
/// value for a bound ONNX leaves out.
pub(super) fn infer_clip(operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
    let [input, min, max] = exact_operands(operands)?;
    Accepts::Numeric.check(input.dtype())?;
    for bound in [min, max] {
        if bound.dtype() != input.dtype() {
            return Err(TypeError::MixedDTypes { lhs: input.dtype(), rhs: bound.dtype() });
        }
        if !bound.shape().dims().is_empty() {
            return Err(TypeError::NotScalar(bound.shape().clone()));
        }
    }
    Ok(vec![TensorType::new(input.dtype(), input.shape().clone(), Layout::RowMajor)])
}

/// `PRelu(x, slope)`: `x` if `x >= 0`, else `slope * x`, where `slope`
/// broadcasts to `x`'s shape (ONNX's unidirectional broadcasting).
pub(super) fn infer_prelu(operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
    let [input, slope] = exact_operands(operands)?;
    Accepts::FloatOrWideInt.check(input.dtype())?;
    if slope.dtype() != input.dtype() {
        return Err(TypeError::MixedDTypes { lhs: input.dtype(), rhs: slope.dtype() });
    }
    broadcast_to(slope.shape(), input.shape())?;
    Ok(vec![TensorType::new(input.dtype(), input.shape().clone(), Layout::RowMajor)])
}
