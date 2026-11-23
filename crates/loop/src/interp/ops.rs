//! The scalar operations, as a GPU computes them: each result rounded or
//! wrapped to its element type.

use super::RunError;
use super::value::{Value, round, saturate, wrap};
use crate::compute::{BinaryOp, InfSign, UnaryOp};
use ptxc_core::DType;
use ptxc_core::float::Overflow;
use std::cmp::Ordering::{self, Equal, Greater, Less};

/// Whether `dtype` is a float type.
const fn is_float(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::Bf16 | DType::F8E4M3Fn | DType::F8E5M2)
}

/// `op` on `value`, of type `dtype`.
pub(super) fn unary(op: UnaryOp, value: Value, dtype: DType) -> Value {
    if op == UnaryOp::Not {
        return Value::Bool(!value.bool());
    }
    if !is_float(dtype) {
        return match op {
            UnaryOp::Abs => wrap(i128::from(value.int()).abs(), dtype),
            UnaryOp::Neg => wrap(-i128::from(value.int()), dtype),
            UnaryOp::Sign => Value::Int(value.int().signum()),
            _ => value,
        };
    }
    let x = value.float();
    let result = match op {
        UnaryOp::IsNaN => return Value::Bool(x.is_nan()),
        UnaryOp::IsInf(sign) => {
            let matches = match sign {
                InfSign::Either => x.is_infinite(),
                InfSign::Positive => x == f64::INFINITY,
                InfSign::Negative => x == f64::NEG_INFINITY,
            };
            return Value::Bool(matches);
        }
        UnaryOp::Abs => x.abs(),
        UnaryOp::Neg => -x,
        UnaryOp::Exp => x.exp(),
        UnaryOp::Expm1 => x.exp_m1(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Log1p => x.ln_1p(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Reciprocal => 1.0 / x,
        UnaryOp::Floor => x.floor(),
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Round => x.round_ties_even(),
        UnaryOp::Sign => sign(x),
        UnaryOp::Erf => libm::erf(x),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Not => x,
    };
    round(result, dtype, Overflow::NonFinite)
}

/// -1, 0 or 1; 0 and NaN are their own sign.
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

/// `op` on `lhs` and `rhs`, both of type `dtype`.
pub(super) fn binary(
    op: BinaryOp,
    lhs: Value,
    rhs: Value,
    dtype: DType,
) -> Result<Value, RunError> {
    if let Some(compared) = compare(op, lhs, rhs, dtype) {
        return Ok(Value::Bool(compared));
    }
    if dtype == DType::Bool {
        let (a, b) = (lhs.bool(), rhs.bool());
        return Ok(Value::Bool(match op {
            BinaryOp::And | BinaryOp::Min => a && b,
            BinaryOp::Or | BinaryOp::Max => a || b,
            _ => a != b,
        }));
    }
    if is_float(dtype) {
        return Ok(round(float_binary(op, lhs.float(), rhs.float()), dtype, Overflow::NonFinite));
    }
    integer_binary(op, i128::from(lhs.int()), i128::from(rhs.int()), dtype)
}

/// A comparison's result, or `None` if `op` is not a comparison.
fn compare(op: BinaryOp, lhs: Value, rhs: Value, dtype: DType) -> Option<bool> {
    let ordering = if is_float(dtype) {
        lhs.float().partial_cmp(&rhs.float())
    } else if dtype == DType::Bool {
        Some(lhs.bool().cmp(&rhs.bool()))
    } else {
        Some(lhs.int().cmp(&rhs.int()))
    };
    let is = |accepted: &[Ordering]| ordering.is_some_and(|ordering| accepted.contains(&ordering));
    match op {
        BinaryOp::Equal => Some(is(&[Equal])),
        BinaryOp::Less => Some(is(&[Less])),
        BinaryOp::LessOrEqual => Some(is(&[Less, Equal])),
        BinaryOp::Greater => Some(is(&[Greater])),
        BinaryOp::GreaterOrEqual => Some(is(&[Greater, Equal])),
        _ => None,
    }
}

fn float_binary(op: BinaryOp, a: f64, b: f64) -> f64 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Pow => a.powf(b),
        BinaryOp::Max => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        }
        BinaryOp::Min => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        }
        _ => a % b,
    }
}

fn integer_binary(op: BinaryOp, a: i128, b: i128, dtype: DType) -> Result<Value, RunError> {
    let divides = || if b == 0 { Err(RunError::DivisionByZero) } else { Ok(()) };
    let result = match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a.wrapping_mul(b),
        BinaryOp::Div => {
            divides()?;
            a / b
        }
        BinaryOp::Max => a.max(b),
        BinaryOp::Min => a.min(b),
        BinaryOp::Pow => i128::from(ptxc_core::arith::wrapping_pow(narrow(a), narrow(b))),
        BinaryOp::FlooredRem => {
            let (a, b) = (narrow(a), narrow(b));
            i128::from(crate::expr::floor_mod(a, b).ok_or(RunError::DivisionByZero)?)
        }
        _ => {
            divides()?;
            a % b
        }
    };
    Ok(wrap(result, dtype))
}

/// An operand of an integer operation, which holds an `i64` it came from.
fn narrow(value: i128) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// `value`, of type `from`, converted to `to`: floats round, integers wrap,
/// floats to integers truncate and saturate (NaN giving 0), and anything
/// becomes a boolean by being non-zero.
pub(super) fn cast(value: Value, from: DType, to: DType, overflow: Overflow) -> Value {
    if to == DType::Bool {
        return Value::Bool(value.bool());
    }
    if is_float(to) {
        return round(value.float(), to, overflow);
    }
    if is_float(from) {
        return saturate(ptxc_core::convert::truncate(value.float()), to);
    }
    wrap(i128::from(value.int()), to)
}
