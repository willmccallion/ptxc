//! Unary elementwise ops, computed in f32 for floats and i64 for integers.

use crate::error::{InterpError, Malformed};
use crate::eval::first;
use crate::numeric::{Elements, store, widen};
use ptxc_core::{HostTensor, TensorData};
use ptxc_graph::{InfDetection, TensorType, UnaryOp};

pub(in crate::eval) fn unary(
    op: UnaryOp,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let rejected = || InterpError::from(Malformed::DType(input.dtype()));
    let elements = match widen(input.data()) {
        Elements::Float(values) => float_unary(op, &values).ok_or_else(rejected)?,
        Elements::Int(values) => {
            let f = int_unary(op).ok_or_else(rejected)?;
            Elements::Int(values.into_iter().map(f).collect())
        }
        Elements::Bool(values) => Elements::Bool(values.into_iter().map(|v| !v).collect()),
    };
    store(elements, output.dtype())
}

/// A float op, `None` for `Not`; `IsNaN` and `IsInf` give bools.
fn float_unary(op: UnaryOp, values: &[f32]) -> Option<Elements> {
    let map = |f: fn(f32) -> f32| Elements::Float(values.iter().map(|&v| f(v)).collect());
    let test = |f: &dyn Fn(f32) -> bool| Elements::Bool(values.iter().map(|&v| f(v)).collect());
    Some(match op {
        UnaryOp::Abs => map(f32::abs),
        UnaryOp::Neg => map(|v| -v),
        UnaryOp::Exp => map(f32::exp),
        UnaryOp::Log => map(f32::ln),
        UnaryOp::Sqrt => map(f32::sqrt),
        UnaryOp::Reciprocal => map(f32::recip),
        UnaryOp::Floor => map(f32::floor),
        UnaryOp::Ceil => map(f32::ceil),
        UnaryOp::Round => map(f32::round_ties_even),
        UnaryOp::Sign => map(sign),
        UnaryOp::Erf => map(libm::erff),
        UnaryOp::Sin => map(f32::sin),
        UnaryOp::Cos => map(f32::cos),
        UnaryOp::IsNaN => test(&f32::is_nan),
        UnaryOp::IsInf(detect) => test(&|v| is_detected_infinity(v, detect)),
        UnaryOp::Not => return None,
    })
}

/// -1, 0 or 1 by sign, and NaN for NaN, as NumPy's `sign`.
fn sign(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        value
    }
}

fn is_detected_infinity(value: f32, detect: InfDetection) -> bool {
    match detect {
        InfDetection::Both => value.is_infinite(),
        InfDetection::Positive => value == f32::INFINITY,
        InfDetection::Negative => value == f32::NEG_INFINITY,
        InfDetection::Neither => false,
    }
}

/// An integer op in i64, `None` for ops only floats take. Narrowing on
/// store wraps, so `Abs` of a type's minimum gives the minimum, as NumPy.
fn int_unary(op: UnaryOp) -> Option<fn(i64) -> i64> {
    match op {
        UnaryOp::Abs => Some(i64::wrapping_abs),
        UnaryOp::Neg => Some(i64::wrapping_neg),
        UnaryOp::Sign => Some(i64::signum),
        _ => None,
    }
}
