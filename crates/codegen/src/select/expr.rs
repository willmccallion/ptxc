//! Selects instructions for index and value expressions.

use super::known::IndexOp;
use super::values::{Held, Narrow, narrow, widen};
use super::{CodegenError, Selector};
use crate::ast::{
    B16, B32, B64, BinaryOp, Compare, CvtMode, F32, Operand, PtxType, Reg, Signedness, Value,
};
use ptxc_core::DType;
use ptxc_core::convert::f64_to_f32;
use ptxc_core::float::Overflow;
use ptxc_loop::{BufferId, Constant, Expr, ExprId, Scalar, ValueExpr};

impl Selector<'_> {
    /// An index expression as an operand: an immediate for a constant.
    pub(super) fn index_operand(&mut self, id: ExprId) -> Result<Value<B64>, CodegenError> {
        match self.kernel.exprs().get(id) {
            Expr::Const(value) => Ok(Value::<B64>::int(value)),
            _ => self.index(id).map(Value::from),
        }
    }

    /// `a op b` on 64-bit index values, as `signedness` reads them: the
    /// register that already holds it where one does. An operation on a
    /// `Var`, whose value changes, is never reused.
    pub(super) fn index_op(
        &mut self,
        op: BinaryOp,
        signedness: Signedness,
        a: Value<B64>,
        b: Value<B64>,
    ) -> Reg<B64> {
        let operands = [a.operand(), b.operand()];
        let fixed = operands.iter().all(|operand| match operand {
            Operand::Reg(reg) => !self.builder.is_variable(*reg),
            Operand::Int(_) | Operand::F32(_) => true,
        });
        let key = IndexOp { op, signedness, operands };
        if let Some(held) = self.known.op(&key) {
            return held;
        }
        let held = self.builder.int(op, signedness, a, b);
        if fixed {
            self.known.remember_op(key, held);
        }
        held
    }

    /// The address of array `name`, from the register that already holds
    /// it where one does.
    pub(super) fn array_address(&mut self, name: &str) -> Reg<B64> {
        if let Some(held) = self.known.array(name) {
            return held;
        }
        let held = self.builder.address_of(name);
        self.known.remember_array(name, held);
        held
    }

    /// Index expression `id` in an `s64` register, computed once a block.
    pub(super) fn index(&mut self, id: ExprId) -> Result<Reg<B64>, CodegenError> {
        if let Some(held) = self.known.index(id) {
            return Ok(held);
        }
        let held = self.computed_index(id)?;
        self.known.remember_index(id, held);
        Ok(held)
    }

    fn computed_index(&mut self, id: ExprId) -> Result<Reg<B64>, CodegenError> {
        let signed = Signedness::Signed;
        let pair = |this: &mut Self, a, b| -> Result<_, CodegenError> {
            Ok((this.index_operand(a)?, this.index_operand(b)?))
        };
        Ok(match self.kernel.exprs().get(id) {
            Expr::Var(var) => *self.vars.get(&var).ok_or(CodegenError::Unbound)?,
            Expr::Const(value) => self.builder.mov(PtxType::S64, Value::<B64>::int(value)),
            Expr::Add(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.index_op(BinaryOp::Add, signed, a, b)
            }
            Expr::Mul(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.index_op(BinaryOp::Mul, signed, a, b)
            }
            Expr::Min(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.index_op(BinaryOp::Min, signed, a, b)
            }
            Expr::Max(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.index_op(BinaryOp::Max, signed, a, b)
            }
            Expr::FloorDiv(a, b) => self.floor_div(a, b)?,
            Expr::FloorMod(a, b) => self.floor_mod(a, b)?,
        })
    }

    /// `a` divided by `b`, floored: an arithmetic shift right where `b` is
    /// a power of two, and 32-bit unsigned division where `a` is known to
    /// fit.
    fn floor_div(&mut self, a: ExprId, b: ExprId) -> Result<Reg<B64>, CodegenError> {
        let dividend = self.index_operand(a)?;
        if let Some(divisor) = self.power_of_two(b) {
            let shift = Value::<B64>::int(i64::from(divisor.trailing_zeros()));
            return Ok(self.index_op(BinaryOp::Shr, Signedness::Signed, dividend, shift));
        }
        if let Some(divisor) = self.unsigned_divisor(a, b) {
            return Ok(self.unsigned32(BinaryOp::Div, dividend, divisor));
        }
        let divisor = self.index_operand(b)?;
        Ok(self.floored(dividend, divisor).0)
    }

    /// The remainder of `a` by `b` with `b`'s sign: a mask where `b` is a
    /// power of two, and a 32-bit unsigned remainder where `a` is known to
    /// fit.
    fn floor_mod(&mut self, a: ExprId, b: ExprId) -> Result<Reg<B64>, CodegenError> {
        let dividend = self.index_operand(a)?;
        if let Some(divisor) = self.power_of_two(b) {
            let mask = Value::<B64>::int(divisor - 1);
            return Ok(self.index_op(BinaryOp::And, Signedness::Bits, dividend, mask));
        }
        if let Some(divisor) = self.unsigned_divisor(a, b) {
            return Ok(self.unsigned32(BinaryOp::Rem, dividend, divisor));
        }
        let divisor = self.index_operand(b)?;
        Ok(self.floored(dividend, divisor).1)
    }

    /// `b`'s value if it is a positive constant that fits in `i32` and every
    /// value `a` takes lies in `0..2^32`: where floored division and modulo
    /// are 32-bit unsigned ones, which ptxas multiplies out.
    fn unsigned_divisor(&self, a: ExprId, b: ExprId) -> Option<i32> {
        let Expr::Const(divisor) = self.kernel.exprs().get(b) else { return None };
        let divisor = i32::try_from(divisor).ok().filter(|&divisor| divisor > 0)?;
        let range = self.kernel.exprs().bounds(a, &self.ranges)?;
        let fits = range.min() >= 0 && u32::try_from(range.max()).is_ok();
        fits.then_some(divisor)
    }

    /// `op`, `Div` or `Rem`, on `dividend`'s low 32 bits and `divisor`,
    /// unsigned, widened back to 64 bits.
    fn unsigned32(&mut self, op: BinaryOp, dividend: Value<B64>, divisor: i32) -> Reg<B64> {
        let narrow: Reg<B32> =
            self.builder.convert(CvtMode::Plain, (PtxType::U32, PtxType::U64), dividend);
        let unsigned = Signedness::Unsigned;
        let result = self.builder.int(op, unsigned, narrow.into(), Value::<B32>::int(divisor));
        self.builder.convert(CvtMode::Plain, (PtxType::U64, PtxType::U32), result.into())
    }

    /// `id`'s value if it is a constant positive power of two. Floored
    /// division by one is an arithmetic shift right and floored modulo a
    /// mask, negative operands included, where `div` and `rem` would be
    /// emulated in software.
    fn power_of_two(&self, id: ExprId) -> Option<i64> {
        let Expr::Const(divisor) = self.kernel.exprs().get(id) else { return None };
        u64::try_from(divisor).is_ok_and(u64::is_power_of_two).then_some(divisor)
    }

    /// `a` divided by `b`, floored, and the remainder with `b`'s sign:
    /// the truncating quotient less one, and the remainder plus `b`, where
    /// the truncating remainder is nonzero with the other sign from `b`.
    fn floored(&mut self, a: Value<B64>, b: Value<B64>) -> (Reg<B64>, Reg<B64>) {
        let signed = Signedness::Signed;
        let quotient = self.index_op(BinaryOp::Div, signed, a, b);
        let remainder = self.index_op(BinaryOp::Rem, signed, a, b);
        let nonzero =
            self.builder.compare_int(Compare::Ne, signed, remainder.into(), Value::<B64>::int(0));
        let signs = self.index_op(BinaryOp::Xor, Signedness::Bits, remainder.into(), b);
        let opposite =
            self.builder.compare_int(Compare::Lt, signed, signs.into(), Value::<B64>::int(0));
        let adjust = self.builder.predicate(BinaryOp::And, nonzero, opposite);
        let lower = self.index_op(BinaryOp::Sub, signed, quotient.into(), Value::<B64>::int(1));
        let shifted = self.index_op(BinaryOp::Add, signed, remainder.into(), b);
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

    /// Value expression `expr`, held in registers, computed once a block.
    pub(super) fn value(&mut self, expr: &ValueExpr) -> Result<Held, CodegenError> {
        if let Some(held) = self.known.value(expr) {
            return Ok(held);
        }
        let held = self.computed_value(expr)?;
        self.known.remember_value(expr, held);
        Ok(held)
    }

    fn computed_value(&mut self, expr: &ValueExpr) -> Result<Held, CodegenError> {
        let buffers = self.kernel.buffers();
        match expr {
            ValueExpr::Load { buffer, index } => self.load(*buffer, *index),
            ValueExpr::Const(scalar) => self.constant(*scalar),
            ValueExpr::Element(buffer) => {
                self.elements.get(buffer).copied().ok_or(CodegenError::Unbound)
            }
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
            ValueExpr::MulAdd { lhs, rhs, addend } => {
                let operands = (self.value(lhs)?, self.value(rhs)?, self.value(addend)?);
                let (Held::Float(lhs), Held::Float(rhs), Held::Float(addend)) = operands else {
                    return Err(CodegenError::Unsupported { op: "a multiply-add off f32" });
                };
                Ok(Held::Float(self.builder.fma(lhs.into(), rhs.into(), addend.into())))
            }
        }
    }

    /// A constant, in the register its type is held in.
    fn constant(&mut self, scalar: Scalar) -> Result<Held, CodegenError> {
        let dtype = scalar.dtype();
        let b = &mut self.builder;
        Ok(match scalar.constant() {
            Constant::Float(value) => {
                Held::Float(b.mov(PtxType::F32, Value::f32(f64_to_f32(value))))
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
        let Some(kind) = Narrow::of(dtype) else { return value };
        let bits = narrow(&mut self.builder, value, kind, overflow);
        self.widened(bits, kind)
    }

    /// `bits`, in the memory form of `kind`, held in f32. The form is kept,
    /// so a store of the result to `kind` reuses `bits`.
    pub(super) fn widened(&mut self, bits: Reg<B16>, kind: Narrow) -> Reg<F32> {
        let wide = widen(&mut self.builder, bits, kind);
        let _new = self.narrow_forms.insert(wide, (bits, kind));
        wide
    }
}
