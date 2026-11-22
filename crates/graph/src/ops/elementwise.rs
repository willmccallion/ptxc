//! Elementwise op kinds and their broadcasting type rule.

use super::{TypeError, TypeRule, exact_operands};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::{DType, Dim, Shape};

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
    /// Logical negation.
    Not,
}

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
    /// `a == b`, giving bool.
    Equal,
    /// `a < b`, giving bool.
    Less,
    /// `a > b`, giving bool.
    Greater,
    /// Logical and.
    And,
    /// Logical or.
    Or,
}

/// The element types an op accepts. FP8 is storage only: it reaches
/// arithmetic through `Cast` or dequantization, as in ONNX.
#[derive(Clone, Copy, Debug)]
enum Accepts {
    /// Floats and integers.
    Numeric,
    /// Floats and signed integers.
    SignedNumeric,
    /// f32, f16 and bf16.
    Float,
    /// Bool only.
    Bool,
    /// Numeric types and bool.
    NumericOrBool,
}

impl Accepts {
    const fn admits(self, dtype: DType) -> bool {
        let float = matches!(dtype, DType::F32 | DType::F16 | DType::Bf16);
        let signed = matches!(dtype, DType::I8 | DType::I32 | DType::I64);
        let unsigned = matches!(dtype, DType::U8);
        let boolean = matches!(dtype, DType::Bool);
        match self {
            Self::Numeric => float || signed || unsigned,
            Self::SignedNumeric => float || signed,
            Self::Float => float,
            Self::Bool => boolean,
            Self::NumericOrBool => float || signed || unsigned || boolean,
        }
    }

    const fn check(self, dtype: DType) -> Result<(), TypeError> {
        if self.admits(dtype) { Ok(()) } else { Err(TypeError::UnsupportedDType(dtype)) }
    }
}

/// The element type of a binary op's result.
#[derive(Clone, Copy, Debug)]
enum ResultDType {
    /// The operands' element type.
    Operands,
    /// Bool.
    Bool,
}

impl UnaryOp {
    const fn accepts(self) -> Accepts {
        match self {
            Self::Abs => Accepts::Numeric,
            Self::Neg => Accepts::SignedNumeric,
            Self::Exp | Self::Log | Self::Sqrt | Self::Reciprocal | Self::Floor | Self::Ceil => {
                Accepts::Float
            }
            Self::Not => Accepts::Bool,
        }
    }
}

impl BinaryOp {
    const fn accepts(self) -> Accepts {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Less | Self::Greater => {
                Accepts::Numeric
            }
            Self::Equal => Accepts::NumericOrBool,
            Self::And | Self::Or => Accepts::Bool,
        }
    }

    const fn result(self) -> ResultDType {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div => ResultDType::Operands,
            Self::Equal | Self::Less | Self::Greater | Self::And | Self::Or => ResultDType::Bool,
        }
    }
}

impl TypeRule for UnaryOp {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        self.accepts().check(input.dtype())?;
        Ok(vec![TensorType::new(input.dtype(), input.shape().clone(), Layout::RowMajor)])
    }
}

impl TypeRule for BinaryOp {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [lhs, rhs] = exact_operands(operands)?;
        if lhs.dtype() != rhs.dtype() {
            return Err(TypeError::MixedDTypes { lhs: lhs.dtype(), rhs: rhs.dtype() });
        }
        self.accepts().check(lhs.dtype())?;
        let shape = broadcast(lhs.shape(), rhs.shape())?;
        let dtype = match self.result() {
            ResultDType::Operands => lhs.dtype(),
            ResultDType::Bool => DType::Bool,
        };
        Ok(vec![TensorType::new(dtype, shape, Layout::RowMajor)])
    }
}

/// ONNX multidirectional broadcasting, as numpy: shapes are aligned at the
/// innermost axis, and each pair of dims must be equal or include a 1.
fn broadcast(lhs: &Shape, rhs: &Shape) -> Result<Shape, TypeError> {
    let rank = lhs.rank().max(rhs.rank()).get();
    let innermost_first: Option<Vec<Dim>> = padded_from_innermost(lhs)
        .zip(padded_from_innermost(rhs))
        .take(rank)
        .map(|(a, b)| broadcast_dim(a, b))
        .collect();
    let Some(mut dims) = innermost_first else {
        return Err(TypeError::NotBroadcastable { lhs: lhs.clone(), rhs: rhs.clone() });
    };
    dims.reverse();
    Ok(Shape::new(dims)?)
}

/// The dims innermost first, then 1s forever, so a lower-rank shape lines up
/// with a higher-rank one.
fn padded_from_innermost(shape: &Shape) -> impl Iterator<Item = Dim> + '_ {
    shape.dims().iter().rev().copied().chain(std::iter::repeat(Dim::new(1)))
}

fn broadcast_dim(a: Dim, b: Dim) -> Option<Dim> {
    let one = Dim::new(1);
    if a == b || b == one {
        Some(a)
    } else if a == one {
        Some(b)
    } else {
        None
    }
}
