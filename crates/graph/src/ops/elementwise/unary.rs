//! Elementwise ops on one tensor.

use crate::ops::accepts::Accepts;
use crate::ops::{TypeError, TypeRule, exact_operands};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::DType;

/// An elementwise op on one tensor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnaryOp {
    /// `|x|`.
    Abs,
    /// `-x`.
    Neg,
    /// `e^x`.
    Exp,
    /// Natural logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// `1 / x`.
    Reciprocal,
    /// Round towards negative infinity.
    Floor,
    /// Round towards positive infinity.
    Ceil,
    /// Round to nearest, ties to even, as ONNX `Round`.
    Round,
    /// -1, 0 or 1 by the sign of `x`.
    Sign,
    /// The error function.
    Erf,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Logical negation.
    Not,
    /// Whether `x` is NaN, giving bool.
    IsNaN,
    /// Whether `x` is an infinity of a detected sign, giving bool.
    IsInf(InfDetection),
}

/// Which infinities ONNX `IsInf` reports, from its `detect_positive` and
/// `detect_negative` attributes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InfDetection {
    /// Both signs.
    Both,
    /// Positive infinity only.
    Positive,
    /// Negative infinity only.
    Negative,
    /// Neither, so every element is false.
    Neither,
}

impl UnaryOp {
    const fn accepts(self) -> Accepts {
        match self {
            Self::Abs | Self::Sign => Accepts::Numeric,
            Self::Neg => Accepts::SignedNumeric,
            Self::Exp
            | Self::Log
            | Self::Sqrt
            | Self::Reciprocal
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Erf
            | Self::Sin
            | Self::Cos
            | Self::IsNaN
            | Self::IsInf(_) => Accepts::Float,
            Self::Not => Accepts::Bool,
        }
    }

    const fn result_dtype(self, operand: DType) -> DType {
        match self {
            Self::IsNaN | Self::IsInf(_) => DType::Bool,
            _ => operand,
        }
    }
}

impl TypeRule for UnaryOp {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        self.accepts().check(input.dtype())?;
        let dtype = self.result_dtype(input.dtype());
        Ok(vec![TensorType::new(dtype, input.shape().clone(), Layout::RowMajor)])
    }
}
