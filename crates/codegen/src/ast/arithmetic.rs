//! `FunctionBuilder`'s arithmetic, comparison and conversion instructions,
//! and the integer register classes they take.

use super::definitions::Var;
use super::function::FunctionBuilder;
use super::instr::{BinaryOp, Compare, CvtMode, Instr, PtxType, UnaryOp};
use super::operand::Value;
use super::reg::{B16, B32, B64, Class, F32, Pred, Reg};

/// An integer register class and the types its operations take.
pub(crate) trait Int: Class {
    /// The width in bits.
    const WIDTH: i64;
    /// The low `WIDTH` bits of `i64::MAX`, as this class reads them signed.
    const LOW_I64_MAX: i64;
    const SIGNED: PtxType;
    const UNSIGNED: PtxType;
    const BITS: PtxType;
}

impl Int for B32 {
    const WIDTH: i64 = 32;
    const LOW_I64_MAX: i64 = -1;
    const SIGNED: PtxType = PtxType::S32;
    const UNSIGNED: PtxType = PtxType::U32;
    const BITS: PtxType = PtxType::B32;
}

impl Int for B64 {
    const WIDTH: i64 = 64;
    const LOW_I64_MAX: i64 = i64::MAX;
    const SIGNED: PtxType = PtxType::S64;
    const UNSIGNED: PtxType = PtxType::U64;
    const BITS: PtxType = PtxType::B64;
}

/// How an integer operation reads its operands.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Signedness {
    Signed,
    Unsigned,
    /// As bits, for bitwise operations.
    Bits,
}

const fn int_type<C: Int>(signedness: Signedness) -> PtxType {
    match signedness {
        Signedness::Signed => C::SIGNED,
        Signedness::Unsigned => C::UNSIGNED,
        Signedness::Bits => C::BITS,
    }
}

impl FunctionBuilder {
    /// A register holding `value`, of the type `ty` names.
    pub(crate) fn mov<C: Class>(&mut self, ty: PtxType, value: Value<C>) -> Reg<C> {
        let dst = self.reg();
        self.push(Instr::Mov { ty, dst: dst.id(), src: value.operand() });
        dst
    }

    /// Sets `dst` to `value`, of the same width, its bits as the type `ty`
    /// names.
    pub(crate) fn assign<From: Class, To: Class>(
        &mut self,
        dst: Var<To>,
        ty: PtxType,
        value: Value<From>,
    ) {
        self.push(Instr::Mov { ty, dst: dst.reg().id(), src: value.operand() });
    }

    pub(crate) fn float(&mut self, op: BinaryOp, a: Value<F32>, b: Value<F32>) -> Reg<F32> {
        let dst = self.reg();
        self.push(Instr::Binary {
            op,
            ty: PtxType::F32,
            dst: dst.id(),
            a: a.operand(),
            b: b.operand(),
        });
        dst
    }

    /// `a * b + c`, rounded once.
    pub(crate) fn fma(&mut self, a: Value<F32>, b: Value<F32>, c: Value<F32>) -> Reg<F32> {
        let dst = self.reg();
        let (a, b, c) = (a.operand(), b.operand(), c.operand());
        self.push(Instr::Fma { dst: dst.id(), a, b, c });
        dst
    }

    pub(crate) fn float_unary(&mut self, op: UnaryOp, a: Value<F32>) -> Reg<F32> {
        let dst = self.reg();
        self.push(Instr::Unary { op, ty: PtxType::F32, dst: dst.id(), a: a.operand() });
        dst
    }

    pub(crate) fn int<C: Int>(
        &mut self,
        op: BinaryOp,
        signedness: Signedness,
        a: Value<C>,
        b: Value<C>,
    ) -> Reg<C> {
        let dst = self.reg();
        let ty = int_type::<C>(signedness);
        self.push(Instr::Binary { op, ty, dst: dst.id(), a: a.operand(), b: b.operand() });
        dst
    }

    pub(crate) fn int_unary<C: Int>(
        &mut self,
        op: UnaryOp,
        signedness: Signedness,
        a: Value<C>,
    ) -> Reg<C> {
        let dst = self.reg();
        let ty = int_type::<C>(signedness);
        self.push(Instr::Unary { op, ty, dst: dst.id(), a: a.operand() });
        dst
    }

    pub(crate) fn predicate(&mut self, op: BinaryOp, a: Reg<Pred>, b: Reg<Pred>) -> Reg<Pred> {
        let dst = self.reg();
        let (a, b) = (Value::from(a).operand(), Value::from(b).operand());
        self.push(Instr::Binary { op, ty: PtxType::Pred, dst: dst.id(), a, b });
        dst
    }

    pub(crate) fn not(&mut self, a: Reg<Pred>) -> Reg<Pred> {
        let dst = self.reg();
        self.push(Instr::Unary {
            op: UnaryOp::Not,
            ty: PtxType::Pred,
            dst: dst.id(),
            a: Value::from(a).operand(),
        });
        dst
    }

    pub(crate) fn compare_float(
        &mut self,
        cmp: Compare,
        a: Value<F32>,
        b: Value<F32>,
    ) -> Reg<Pred> {
        let dst = self.reg();
        self.push(Instr::Setp {
            cmp,
            ty: PtxType::F32,
            dst: dst.id(),
            a: a.operand(),
            b: b.operand(),
        });
        dst
    }

    pub(crate) fn compare_int<C: Int>(
        &mut self,
        cmp: Compare,
        signedness: Signedness,
        a: Value<C>,
        b: Value<C>,
    ) -> Reg<Pred> {
        let dst = self.reg();
        let ty = int_type::<C>(signedness);
        self.push(Instr::Setp { cmp, ty, dst: dst.id(), a: a.operand(), b: b.operand() });
        dst
    }

    /// `a` where `pred` holds, else `b`.
    pub(crate) fn select<C: Class>(
        &mut self,
        ty: PtxType,
        a: Value<C>,
        b: Value<C>,
        pred: Reg<Pred>,
    ) -> Reg<C> {
        let dst = self.reg();
        self.push(Instr::Selp {
            ty,
            dst: dst.id(),
            a: a.operand(),
            b: b.operand(),
            pred: pred.id(),
        });
        dst
    }

    /// `src` converted from the type `from` names to the type `to` names.
    pub(crate) fn convert<From: Class, To: Class>(
        &mut self,
        mode: CvtMode,
        (to, from): (PtxType, PtxType),
        src: Value<From>,
    ) -> Reg<To> {
        let dst = self.reg();
        self.push(Instr::Cvt { mode, to, from, dst: dst.id(), src: vec![src.operand()] });
        dst
    }

    /// Two f32 values rounded into an FP8 pair of type `to`, `high` in the
    /// high byte.
    pub(crate) fn pack_fp8(
        &mut self,
        to: PtxType,
        (high, low): (Value<F32>, Value<F32>),
    ) -> Reg<B16> {
        let dst = self.reg();
        let src = vec![high.operand(), low.operand()];
        self.push(Instr::Cvt {
            mode: CvtMode::RnSatfinite,
            to,
            from: PtxType::F32,
            dst: dst.id(),
            src,
        });
        dst
    }

    /// The bits of `value` as another class of the same size.
    pub(crate) fn reinterpret<From: Class, To: Class>(
        &mut self,
        ty: PtxType,
        value: Reg<From>,
    ) -> Reg<To> {
        let dst = self.reg();
        self.push(Instr::Mov { ty, dst: dst.id(), src: Value::from(value).operand() });
        dst
    }
}
