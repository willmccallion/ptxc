//! Selects instructions for index and value expressions.

use super::values::{Held, round_float};
use super::{CodegenError, Selector};
use crate::ast::{B32, B64, BinaryOp, Compare, CvtMode, F32, PtxType, Reg, Signedness, Value};
use ptxc_core::DType;
use ptxc_core::convert::f64_to_f32;
use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16, Overflow};
use ptxc_loop::{BufferId, Constant, Expr, ExprId, Scalar, ValueExpr};

impl Selector<'_> {
    /// An index expression as an operand: an immediate for a constant.
    pub(super) fn index_operand(&mut self, id: ExprId) -> Result<Value<B64>, CodegenError> {
        match self.kernel.exprs().get(id) {
            Expr::Const(value) => Ok(Value::<B64>::int(value)),
            _ => self.index(id).map(Value::from),
        }
    }

    /// Index expression `id` in an `s64` register.
    pub(super) fn index(&mut self, id: ExprId) -> Result<Reg<B64>, CodegenError> {
        let signed = Signedness::Signed;
        let pair = |this: &mut Self, a, b| -> Result<_, CodegenError> {
            Ok((this.index_operand(a)?, this.index_operand(b)?))
        };
        Ok(match self.kernel.exprs().get(id) {
            Expr::Var(var) => *self.vars.get(&var).ok_or(CodegenError::Unbound)?,
            Expr::Const(value) => self.builder.mov(PtxType::S64, Value::<B64>::int(value)),
            Expr::Add(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.builder.int(BinaryOp::Add, signed, a, b)
            }
            Expr::Mul(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.builder.int(BinaryOp::Mul, signed, a, b)
            }
            Expr::Min(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.builder.int(BinaryOp::Min, signed, a, b)
            }
            Expr::Max(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.builder.int(BinaryOp::Max, signed, a, b)
            }
            Expr::FloorDiv(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.floored(a, b).0
            }
            Expr::FloorMod(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.floored(a, b).1
            }
        })
    }

    /// `a` divided by `b`, floored, and the remainder with `b`'s sign:
    /// the truncating quotient less one, and the remainder plus `b`, where
    /// the truncating remainder is nonzero with the other sign from `b`.
    fn floored(&mut self, a: Value<B64>, b: Value<B64>) -> (Reg<B64>, Reg<B64>) {
        let signed = Signedness::Signed;
        let quotient = self.builder.int(BinaryOp::Div, signed, a, b);
        let remainder = self.builder.int(BinaryOp::Rem, signed, a, b);
        let nonzero =
            self.builder.compare_int(Compare::Ne, signed, remainder.into(), Value::<B64>::int(0));
        let signs = self.builder.int(BinaryOp::Xor, Signedness::Bits, remainder.into(), b);
        let opposite =
            self.builder.compare_int(Compare::Lt, signed, signs.into(), Value::<B64>::int(0));
        let adjust = self.builder.predicate(BinaryOp::And, nonzero, opposite);
        let lower = self.builder.int(BinaryOp::Sub, signed, quotient.into(), Value::<B64>::int(1));
        let shifted = self.builder.int(BinaryOp::Add, signed, remainder.into(), b);
        let quotient = self.builder.select(PtxType::B64, lower.into(), quotient.into(), adjust);
        let remainder = self.builder.select(PtxType::B64, shifted.into(), remainder.into(), adjust);
        (quotient, remainder)
    }

    /// The integer element `index` of `buffer`, an index tensor, as `s64`.
    pub(super) fn load_index(
        &mut self,
        buffer: BufferId,
        index: ExprId,
    ) -> Result<Reg<B64>, CodegenError> {
        Ok(match self.load(buffer, index)? {
            Held::Int64(value) => value,
            Held::Int32(value) => {
                self.builder.convert(CvtMode::Plain, (PtxType::S64, PtxType::S32), value.into())
            }
            Held::Float(_) | Held::Bool(_) => {
                return Err(CodegenError::Unsupported { op: "non-integer index" });
            }
        })
    }

    /// Value expression `expr`, held in registers.
    pub(super) fn value(&mut self, expr: &ValueExpr) -> Result<Held, CodegenError> {
        let buffers = self.kernel.buffers();
        match expr {
            ValueExpr::Load { buffer, index } => self.load(*buffer, *index),
            ValueExpr::Const(scalar) => self.constant(*scalar),
            ValueExpr::Index { index, dtype } => {
                let position = self.index(*index)?;
                Ok(self.integer_from_index(position, *dtype))
            }
            ValueExpr::Unary(op, operand) => {
                let dtype = operand.dtype(buffers);
                let value = self.value(operand)?;
                self.unary(*op, value, dtype)
            }
            ValueExpr::Binary(op, lhs, rhs) => {
                let dtype = lhs.dtype(buffers);
                let (a, b) = (self.value(lhs)?, self.value(rhs)?);
                self.binary(*op, (a, b), dtype)
            }
            ValueExpr::Cast { to, overflow, value } => {
                let from = value.dtype(buffers);
                let value = self.value(value)?;
                Ok(self.cast(value, (from, *to), *overflow))
            }
            ValueExpr::Select { condition, then, otherwise } => {
                let condition = self.value(condition)?;
                let (then, otherwise) = (self.value(then)?, self.value(otherwise)?);
                self.select(condition, then, otherwise)
            }
        }
    }

    /// A constant, in the register its type is held in.
    fn constant(&mut self, scalar: Scalar) -> Result<Held, CodegenError> {
        let dtype = scalar.dtype();
        let b = &mut self.builder;
        Ok(match scalar.constant() {
            Constant::Float(value) => {
                Held::Float(b.mov(PtxType::F32, Value::f32(rounded_constant(value, dtype))))
            }
            Constant::Bool(value) => {
                let bit = b.mov(PtxType::B32, Value::<B32>::int(i32::from(value)));
                let zero = Value::<B32>::int(0);
                Held::Bool(b.compare_int(Compare::Ne, Signedness::Unsigned, bit.into(), zero))
            }
            Constant::Int(value) if dtype == DType::I64 => {
                Held::Int64(b.mov(PtxType::S64, Value::<B64>::int(value)))
            }
            Constant::Int(value) => {
                // An 8- or 32-bit constant fits its type, so it fits 32 bits.
                let narrow = i32::try_from(value).map_err(|_| CodegenError::TooLarge)?;
                Held::Int32(b.mov(PtxType::S32, Value::<B32>::int(narrow)))
            }
        })
    }

    /// `position` as an integer of `dtype`, wrapping.
    fn integer_from_index(&mut self, position: Reg<B64>, dtype: DType) -> Held {
        if dtype == DType::I64 {
            return Held::Int64(position);
        }
        let low: Reg<B32> =
            self.builder.convert(CvtMode::Plain, (PtxType::U32, PtxType::U64), position.into());
        Held::Int32(super::values::wrap32(&mut self.builder, low, dtype))
    }

    /// `then` where `condition` holds, else `otherwise`.
    fn select(
        &mut self,
        condition: Held,
        then: Held,
        otherwise: Held,
    ) -> Result<Held, CodegenError> {
        let Held::Bool(pred) = condition else {
            return Err(CodegenError::Unsupported { op: "non-boolean select" });
        };
        let b = &mut self.builder;
        Ok(match (then, otherwise) {
            (Held::Float(a), Held::Float(c)) => {
                Held::Float(b.select(PtxType::F32, a.into(), c.into(), pred))
            }
            (Held::Int32(a), Held::Int32(c)) => {
                Held::Int32(b.select(PtxType::B32, a.into(), c.into(), pred))
            }
            (Held::Int64(a), Held::Int64(c)) => {
                Held::Int64(b.select(PtxType::B64, a.into(), c.into(), pred))
            }
            (Held::Bool(a), Held::Bool(c)) => {
                let taken = b.predicate(BinaryOp::And, pred, a);
                let other = b.not(pred);
                let skipped = b.predicate(BinaryOp::And, other, c);
                Held::Bool(b.predicate(BinaryOp::Or, taken, skipped))
            }
            _ => return Err(CodegenError::Unsupported { op: "select of mixed types" }),
        })
    }

    /// A float `value` rounded to `dtype` and held in f32, FP8 overflow
    /// following `overflow`.
    pub(super) fn rounded(
        &mut self,
        value: Reg<F32>,
        dtype: DType,
        overflow: Overflow,
    ) -> Reg<F32> {
        round_float(&mut self.builder, value, dtype, overflow)
    }
}

/// A float constant's value in `dtype`, as the f32 that holds it exactly.
fn rounded_constant(value: f64, dtype: DType) -> f32 {
    let narrowed = match dtype {
        DType::F16 => F16::from_f64(value).to_f64(),
        DType::Bf16 => Bf16::from_f64(value).to_f64(),
        DType::F8E4M3Fn => F8E4M3Fn::from_f64(value, Overflow::NonFinite).to_f64(),
        DType::F8E5M2 => F8E5M2::from_f64(value, Overflow::NonFinite).to_f64(),
        DType::F32 | DType::I8 | DType::U8 | DType::I32 | DType::I64 | DType::Bool => value,
    };
    f64_to_f32(narrowed)
}
