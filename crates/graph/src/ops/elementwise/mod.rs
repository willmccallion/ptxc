//! Elementwise ops and ONNX's multidirectional broadcasting.

mod binary;
mod cast;
mod unary;
mod variadic;

pub use binary::{BinaryOp, ModKind};
pub use cast::CastAttrs;
pub use unary::{InfDetection, UnaryOp};
pub use variadic::VariadicOp;
pub(in crate::ops) use variadic::infer_where;

use super::TypeError;
use crate::tensor_type::TensorType;
use ptxc_core::{DType, Dim, Shape};

/// The broadcast shape of every operand, which must share one element type.
fn broadcast_same_dtype(operands: &[&TensorType]) -> Result<(DType, Shape), TypeError> {
    let Some((first, rest)) = operands.split_first() else {
        return Err(TypeError::NoOperands);
    };
    let mut shape = first.shape().clone();
    for operand in rest {
        if operand.dtype() != first.dtype() {
            return Err(TypeError::MixedDTypes { lhs: first.dtype(), rhs: operand.dtype() });
        }
        shape = broadcast(&shape, operand.shape())?;
    }
    Ok((first.dtype(), shape))
}

/// ONNX multidirectional broadcasting, as numpy: shapes are aligned at the
/// innermost axis, and each pair of dims must be equal or include a 1.
pub(in crate::ops) fn broadcast(lhs: &Shape, rhs: &Shape) -> Result<Shape, TypeError> {
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
