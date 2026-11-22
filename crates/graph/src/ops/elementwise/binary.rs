//! Elementwise ops on two tensors, with broadcasting.

use super::broadcast;
use crate::ops::accepts::Accepts;
use crate::ops::{TypeError, TypeRule, exact_operands};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::DType;

/// An elementwise op on two tensors.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinaryOp {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
    /// `a / b`; integer division truncates, as ONNX `Div`.
    Div,
    /// `a ^ b`. The exponent may have a different element type from the
    /// base; the result has the base's, as ONNX `Pow`.
    Pow,
    /// The remainder of `a / b`.
    Mod(ModKind),
    /// `a == b`, giving bool.
    Equal,
    /// `a < b`, giving bool.
    Less,
    /// `a <= b`, giving bool.
    LessOrEqual,
    /// `a > b`, giving bool.
    Greater,
    /// `a >= b`, giving bool.
    GreaterOrEqual,
    /// Logical and.
    And,
    /// Logical or.
    Or,
    /// Logical exclusive or.
    Xor,
}

/// How ONNX `Mod` takes the sign of its result, from its `fmod` attribute.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ModKind {
    /// `fmod = 0`: the sign of the divisor, as Python's `%`. Integers only.
    Integer,
    /// `fmod = 1`: the sign of the dividend, as C's `fmod`.
    Fmod,
}

/// The element type of a binary op's result.
#[derive(Clone, Copy, Debug)]
enum ResultDType {
    /// The first operand's element type.
    Lhs,
    /// Bool.
    Bool,
}

impl BinaryOp {
    const fn accepts(self) -> Accepts {
        match self {
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Mod(ModKind::Fmod)
            | Self::Less
            | Self::LessOrEqual
            | Self::Greater
            | Self::GreaterOrEqual => Accepts::Numeric,
            Self::Mod(ModKind::Integer) => Accepts::Integer,
            Self::Pow => Accepts::FloatOrWideInt,
            Self::Equal => Accepts::NumericOrBool,
            Self::And | Self::Or | Self::Xor => Accepts::Bool,
        }
    }

    const fn result(self) -> ResultDType {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Pow | Self::Mod(_) => {
                ResultDType::Lhs
            }
            Self::Equal
            | Self::Less
            | Self::LessOrEqual
            | Self::Greater
            | Self::GreaterOrEqual
            | Self::And
            | Self::Or
            | Self::Xor => ResultDType::Bool,
        }
    }

    /// Checks the operands' element types: `Pow`'s exponent is any numeric
    /// type, and every other op's operands share one type.
    fn check_dtypes(self, lhs: DType, rhs: DType) -> Result<(), TypeError> {
        self.accepts().check(lhs)?;
        match self {
            Self::Pow => Accepts::Numeric.check(rhs),
            _ if lhs != rhs => Err(TypeError::MixedDTypes { lhs, rhs }),
            _ => Ok(()),
        }
    }
}

impl TypeRule for BinaryOp {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [lhs, rhs] = exact_operands(operands)?;
        self.check_dtypes(lhs.dtype(), rhs.dtype())?;
        let shape = broadcast(lhs.shape(), rhs.shape())?;
        let dtype = match self.result() {
            ResultDType::Lhs => lhs.dtype(),
            ResultDType::Bool => DType::Bool,
        };
        Ok(vec![TensorType::new(dtype, shape, Layout::RowMajor)])
    }
}
