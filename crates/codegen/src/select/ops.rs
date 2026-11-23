//! Selects the scalar operations: unary and binary operations and casts,
//! each computed in the register its type is held in and rounded or
//! wrapped back to that type.

use super::values::{Held, wrap32};
use super::{CodegenError, Selector};
use crate::ast::{
    B32, B64, BinaryOp as Op, Compare, CvtMode, F32, Int, Pred, PtxType, Reg, Signedness,
    UnaryOp as PtxUnary, Value,
};
use ptxc_core::DType;
use ptxc_core::float::Overflow;
use ptxc_loop::{BinaryOp, InfSign, UnaryOp};

const fn unsupported(op: &'static str) -> CodegenError {
    CodegenError::Unsupported { op }
}

const fn signedness(dtype: DType) -> Signedness {
    if matches!(dtype, DType::U8) { Signedness::Unsigned } else { Signedness::Signed }
}

impl Selector<'_> {
    pub(super) fn unary(
        &mut self,
        op: UnaryOp,
        value: Held,
        dtype: DType,
    ) -> Result<Held, CodegenError> {
        match value {
            Held::Float(x) => self.float_unary(op, x, dtype),
            Held::Int32(x) => {
                let result = self.int_unary(op, x)?;
                Ok(Held::Int32(wrap32(&mut self.builder, result, dtype)))
            }
            Held::Int64(x) => Ok(Held::Int64(self.int_unary(op, x)?)),
            Held::Bool(x) if op == UnaryOp::Not => Ok(Held::Bool(self.builder.not(x))),
            Held::Bool(_) => Err(unsupported(op.name())),
        }
    }

    fn float_unary(
        &mut self,
        op: UnaryOp,
        x: Reg<F32>,
        dtype: DType,
    ) -> Result<Held, CodegenError> {
        let b = &mut self.builder;
        let integral = |b: &mut crate::ast::FunctionBuilder, mode| {
            b.convert::<F32, F32>(mode, (PtxType::F32, PtxType::F32), x.into())
        };
        let result = match op {
            UnaryOp::Abs => b.float_unary(PtxUnary::Abs, x.into()),
            UnaryOp::Neg => b.float_unary(PtxUnary::Neg, x.into()),
            UnaryOp::Sqrt => b.float_unary(PtxUnary::Sqrt, x.into()),
            UnaryOp::Reciprocal => b.float_unary(PtxUnary::Rcp, x.into()),
            UnaryOp::Floor => integral(b, CvtMode::Rmi),
            UnaryOp::Ceil => integral(b, CvtMode::Rpi),
            UnaryOp::Round => integral(b, CvtMode::Rni),
            UnaryOp::Sign => {
                let positive = b.compare_float(Compare::Gt, x.into(), Value::f32(0.0));
                let negative = b.compare_float(Compare::Lt, x.into(), Value::f32(0.0));
                let below = b.select(PtxType::F32, Value::f32(-1.0), x.into(), negative);
                b.select(PtxType::F32, Value::f32(1.0), below.into(), positive)
            }
            UnaryOp::IsNaN => {
                return Ok(Held::Bool(b.compare_float(Compare::Unordered, x.into(), x.into())));
            }
            UnaryOp::IsInf(sign) => return Ok(Held::Bool(self.is_infinite(x, sign))),
            UnaryOp::Not
            | UnaryOp::Exp
            | UnaryOp::Expm1
            | UnaryOp::Log
            | UnaryOp::Log1p
            | UnaryOp::Erf
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Tanh => return Err(unsupported(op.name())),
        };
        Ok(Held::Float(self.rounded(result, dtype, Overflow::NonFinite)))
    }

    fn is_infinite(&mut self, x: Reg<F32>, sign: InfSign) -> Reg<Pred> {
        let b = &mut self.builder;
        let infinity = Value::f32(f32::INFINITY);
        match sign {
            InfSign::Positive => b.compare_float(Compare::Eq, x.into(), infinity),
            InfSign::Negative => {
                b.compare_float(Compare::Eq, x.into(), Value::f32(f32::NEG_INFINITY))
            }
            InfSign::Either => {
                let magnitude = b.float_unary(PtxUnary::Abs, x.into());
                b.compare_float(Compare::Eq, magnitude.into(), infinity)
            }
        }
    }

    fn int_unary<C: Int>(&mut self, op: UnaryOp, x: Reg<C>) -> Result<Reg<C>, CodegenError> {
        let b = &mut self.builder;
        let signed = Signedness::Signed;
        Ok(match op {
            UnaryOp::Abs => b.int_unary(PtxUnary::Abs, signed, x.into()),
            UnaryOp::Neg => b.int_unary(PtxUnary::Neg, signed, x.into()),
            UnaryOp::Sign => {
                let zero = Value::<C>::immediate(0);
                let positive = b.compare_int(Compare::Gt, signed, x.into(), zero);
                let negative = b.compare_int(Compare::Lt, signed, x.into(), zero);
                let below = b.select(C::BITS, Value::<C>::immediate(-1), zero, negative);
                b.select(C::BITS, Value::<C>::immediate(1), below.into(), positive)
            }
            _ => return Err(unsupported(op.name())),
        })
    }

    pub(super) fn binary(
        &mut self,
        op: BinaryOp,
        (a, b): (Held, Held),
        dtype: DType,
    ) -> Result<Held, CodegenError> {
        match (a, b) {
            (Held::Float(a), Held::Float(b)) => self.float_binary(op, (a, b), dtype),
            (Held::Int32(a), Held::Int32(b)) => {
                match self.int_binary(op, (a, b), signedness(dtype))? {
                    Held::Int32(result) => {
                        Ok(Held::Int32(wrap32(&mut self.builder, result, dtype)))
                    }
                    other => Ok(other),
                }
            }
            (Held::Int64(a), Held::Int64(b)) => self.int_binary(op, (a, b), Signedness::Signed),
            (Held::Bool(a), Held::Bool(b)) => self.bool_binary(op, a, b),
            _ => Err(unsupported("binary operation on mixed types")),
        }
    }

    fn float_binary(
        &mut self,
        op: BinaryOp,
        (x, y): (Reg<F32>, Reg<F32>),
        dtype: DType,
    ) -> Result<Held, CodegenError> {
        let b = &mut self.builder;
        let compare = |b: &mut crate::ast::FunctionBuilder, cmp| {
            Held::Bool(b.compare_float(cmp, x.into(), y.into()))
        };
        let result = match op {
            BinaryOp::Add => b.float(Op::Add, x.into(), y.into()),
            BinaryOp::Sub => b.float(Op::Sub, x.into(), y.into()),
            BinaryOp::Mul => b.float(Op::Mul, x.into(), y.into()),
            BinaryOp::Div => b.float(Op::Div, x.into(), y.into()),
            BinaryOp::Max => b.float(Op::Max, x.into(), y.into()),
            BinaryOp::Min => b.float(Op::Min, x.into(), y.into()),
            BinaryOp::Equal => return Ok(compare(b, Compare::Eq)),
            BinaryOp::Less => return Ok(compare(b, Compare::Lt)),
            BinaryOp::LessOrEqual => return Ok(compare(b, Compare::Le)),
            BinaryOp::Greater => return Ok(compare(b, Compare::Gt)),
            BinaryOp::GreaterOrEqual => return Ok(compare(b, Compare::Ge)),
            BinaryOp::Pow
            | BinaryOp::TruncatedRem
            | BinaryOp::FlooredRem
            | BinaryOp::And
            | BinaryOp::Or
            | BinaryOp::Xor => {
                return Err(unsupported(op.name()));
            }
        };
        Ok(Held::Float(self.rounded(result, dtype, Overflow::NonFinite)))
    }

    fn int_binary<C: Int>(
        &mut self,
        op: BinaryOp,
        (x, y): (Reg<C>, Reg<C>),
        signedness: Signedness,
    ) -> Result<Held, CodegenError>
    where
        Held: From<Reg<C>>,
    {
        let b = &mut self.builder;
        let arithmetic = |b: &mut crate::ast::FunctionBuilder, op| {
            Held::from(b.int(op, signedness, x.into(), y.into()))
        };
        let compare = |b: &mut crate::ast::FunctionBuilder, cmp| {
            Held::Bool(b.compare_int(cmp, signedness, x.into(), y.into()))
        };
        Ok(match op {
            BinaryOp::Add => arithmetic(b, Op::Add),
            BinaryOp::Sub => arithmetic(b, Op::Sub),
            BinaryOp::Mul => arithmetic(b, Op::Mul),
            BinaryOp::Div => arithmetic(b, Op::Div),
            BinaryOp::TruncatedRem => arithmetic(b, Op::Rem),
            BinaryOp::Max => arithmetic(b, Op::Max),
            BinaryOp::Min => arithmetic(b, Op::Min),
            BinaryOp::FlooredRem => {
                let remainder = b.int(Op::Rem, signedness, x.into(), y.into());
                let zero = Value::<C>::immediate(0);
                let nonzero = b.compare_int(Compare::Ne, signedness, remainder.into(), zero);
                let signs = b.int(Op::Xor, Signedness::Bits, remainder.into(), y.into());
                let opposite = b.compare_int(Compare::Lt, Signedness::Signed, signs.into(), zero);
                let adjust = b.predicate(Op::And, nonzero, opposite);
                let shifted = b.int(Op::Add, signedness, remainder.into(), y.into());
                Held::from(b.select(C::BITS, shifted.into(), remainder.into(), adjust))
            }
            BinaryOp::Equal => compare(b, Compare::Eq),
            BinaryOp::Less => compare(b, Compare::Lt),
            BinaryOp::LessOrEqual => compare(b, Compare::Le),
            BinaryOp::Greater => compare(b, Compare::Gt),
            BinaryOp::GreaterOrEqual => compare(b, Compare::Ge),
            BinaryOp::Pow | BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => {
                return Err(unsupported(op.name()));
            }
        })
    }

    fn bool_binary(
        &mut self,
        op: BinaryOp,
        x: Reg<Pred>,
        y: Reg<Pred>,
    ) -> Result<Held, CodegenError> {
        let b = &mut self.builder;
        Ok(Held::Bool(match op {
            BinaryOp::And | BinaryOp::Min => b.predicate(Op::And, x, y),
            BinaryOp::Or | BinaryOp::Max => b.predicate(Op::Or, x, y),
            BinaryOp::Xor => b.predicate(Op::Xor, x, y),
            BinaryOp::Equal => {
                let different = b.predicate(Op::Xor, x, y);
                b.not(different)
            }
            _ => return Err(unsupported(op.name())),
        }))
    }
}

impl From<Reg<B32>> for Held {
    fn from(reg: Reg<B32>) -> Self {
        Self::Int32(reg)
    }
}

impl From<Reg<B64>> for Held {
    fn from(reg: Reg<B64>) -> Self {
        Self::Int64(reg)
    }
}
