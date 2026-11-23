//! Binary elementwise ops with ONNX's multidirectional broadcasting.

use crate::error::{InterpError, Malformed};
use crate::extents::{Extents, broadcast_map};
use crate::numeric::{Elements, int_to_f32, int_to_f64, store, truncate, widen};
use ptxc_core::{DType, HostTensor, TensorData};
use ptxc_graph::{BinaryOp, ModKind, TensorType};

/// Two operands' elements broadcast to the output.
struct Pair<'a, T> {
    lhs: (&'a [T], &'a Extents),
    rhs: (&'a [T], &'a Extents),
    out: &'a Extents,
}

impl<T: Copy> Pair<'_, T> {
    fn map<U>(&self, f: impl Fn(T, T) -> U) -> Vec<U> {
        broadcast_map(&[self.lhs, self.rhs], self.out, |args| f(args[0], args[1]))
    }
}

pub(in crate::eval) fn binary(
    op: BinaryOp,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let [lhs, rhs] = operands else { return Err(Malformed::Operands.into()) };
    let extents =
        (Extents::of(lhs.shape())?, Extents::of(rhs.shape())?, Extents::of(output.shape())?);
    let (l, r, out) = (&extents.0, &extents.1, &extents.2);
    let rejected = InterpError::from(Malformed::DType(rhs.dtype()));
    let elements = match (widen(lhs.data()), widen(rhs.data())) {
        (Elements::Float(a), Elements::Float(b)) => {
            float_binary(op, &Pair { lhs: (&a, l), rhs: (&b, r), out }, lhs.dtype())?
        }
        (Elements::Float(a), Elements::Int(b)) => {
            let b: Vec<f32> = b.into_iter().map(int_to_f32).collect();
            float_binary(op, &Pair { lhs: (&a, l), rhs: (&b, r), out }, lhs.dtype())?
        }
        (Elements::Int(a), Elements::Int(b)) => {
            int_binary(op, &Pair { lhs: (&a, l), rhs: (&b, r), out }, lhs.dtype())?
        }
        (Elements::Int(a), Elements::Float(b)) => {
            let a: Vec<f64> = a.into_iter().map(int_to_f64).collect();
            let b: Vec<f64> = b.into_iter().map(f64::from).collect();
            Elements::Int(Pair { lhs: (&a, l), rhs: (&b, r), out }.map(|x, y| truncate(x.powf(y))))
        }
        (Elements::Bool(a), Elements::Bool(b)) => {
            bool_binary(op, &Pair { lhs: (&a, l), rhs: (&b, r), out }).ok_or(rejected)?
        }
        _ => return Err(rejected),
    };
    store(elements, output.dtype())
}

/// Comparisons, which every domain shares and which give bools.
fn compare<T: PartialOrd + Copy>(op: BinaryOp, pair: &Pair<'_, T>) -> Option<Elements> {
    let test: fn(&T, &T) -> bool = match op {
        BinaryOp::Equal => T::eq,
        BinaryOp::Less => T::lt,
        BinaryOp::LessOrEqual => T::le,
        BinaryOp::Greater => T::gt,
        BinaryOp::GreaterOrEqual => T::ge,
        _ => return None,
    };
    Some(Elements::Bool(pair.map(|a, b| test(&a, &b))))
}

/// Float arithmetic in f32; `Mod` with `fmod` takes the dividend's sign.
fn float_binary(op: BinaryOp, pair: &Pair<'_, f32>, dtype: DType) -> Result<Elements, InterpError> {
    if let Some(compared) = compare(op, pair) {
        return Ok(compared);
    }
    let f: fn(f32, f32) -> f32 = match op {
        BinaryOp::Add => |a, b| a + b,
        BinaryOp::Sub => |a, b| a - b,
        BinaryOp::Mul => |a, b| a * b,
        BinaryOp::Div => |a, b| a / b,
        BinaryOp::Pow => f32::powf,
        BinaryOp::Mod(ModKind::Fmod) => |a, b| a % b,
        _ => return Err(Malformed::DType(dtype).into()),
    };
    Ok(Elements::Float(pair.map(f)))
}

/// Integer arithmetic in i64, wrapping; division truncates toward zero.
fn int_binary(op: BinaryOp, pair: &Pair<'_, i64>, dtype: DType) -> Result<Elements, InterpError> {
    if let Some(compared) = compare(op, pair) {
        return Ok(compared);
    }
    let f: fn(i64, i64) -> Option<i64> = match op {
        BinaryOp::Add => |a, b| Some(a.wrapping_add(b)),
        BinaryOp::Sub => |a, b| Some(a.wrapping_sub(b)),
        BinaryOp::Mul => |a, b| Some(a.wrapping_mul(b)),
        BinaryOp::Div => |a, b| (b != 0).then(|| a.wrapping_div(b)),
        BinaryOp::Mod(ModKind::Fmod) => |a, b| (b != 0).then(|| a.wrapping_rem(b)),
        BinaryOp::Mod(ModKind::Integer) => floored_rem,
        BinaryOp::Pow => |a, b| Some(ptxc_core::arith::wrapping_pow(a, b)),
        _ => return Err(Malformed::DType(dtype).into()),
    };
    let values: Option<Vec<i64>> = pair.map(f).into_iter().collect();
    let op_name = if matches!(op, BinaryOp::Div) { "Div" } else { "Mod" };
    Ok(Elements::Int(values.ok_or(InterpError::IntegerDivisionByZero { op: op_name })?))
}

/// The remainder with the divisor's sign, as Python's `%`.
fn floored_rem(a: i64, b: i64) -> Option<i64> {
    let remainder = a.checked_rem(b).or_else(|| (b != 0).then_some(0))?;
    Some(if remainder != 0 && remainder.signum() != b.signum() { remainder + b } else { remainder })
}

fn bool_binary(op: BinaryOp, pair: &Pair<'_, bool>) -> Option<Elements> {
    let f: fn(bool, bool) -> bool = match op {
        BinaryOp::Equal => |a, b| a == b,
        BinaryOp::And => |a, b| a && b,
        BinaryOp::Or => |a, b| a || b,
        BinaryOp::Xor => |a, b| a != b,
        _ => return None,
    };
    Some(Elements::Bool(pair.map(f)))
}
