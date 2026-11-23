//! Emits a math program (D51) inline, each step as the one PTX instruction
//! whose IEEE or integer result the loop interpreter computes for it.

use super::Selector;
use crate::ast::{
    B32, BinaryOp, Compare, CvtMode, F32, FunctionBuilder, Pred, PtxType, Reg, Signedness, Value,
};
use ptxc_core::IndexVec;
use ptxc_loop::math::{
    Comparison, FloatDef, FloatOp, FloatSlot, IntDef, IntOp, IntSlot, MathProgram, PredDef,
    PredSlot, Step,
};

/// The registers holding the slots a program has defined so far.
struct Slots {
    floats: IndexVec<FloatSlot, Reg<F32>>,
    ints: IndexVec<IntSlot, Reg<B32>>,
    preds: IndexVec<PredSlot, Reg<Pred>>,
}

impl Selector<'_> {
    /// `program` applied to `argument`.
    pub(super) fn math(&mut self, program: &MathProgram, argument: Reg<F32>) -> Reg<F32> {
        let mut slots =
            Slots { floats: IndexVec::new(), ints: IndexVec::new(), preds: IndexVec::new() };
        let code = &mut self.builder;
        for step in program.steps() {
            match *step {
                Step::Float(def) => {
                    let reg = slots.float(code, def, argument);
                    let _slot = slots.floats.push(reg);
                }
                Step::Int(def) => {
                    let reg = slots.int(code, def);
                    let _slot = slots.ints.push(reg);
                }
                Step::Pred(def) => {
                    let reg = slots.pred(code, def);
                    let _slot = slots.preds.push(reg);
                }
            }
        }
        slots.floats[program.result()]
    }
}

impl Slots {
    fn float(&self, code: &mut FunctionBuilder, def: FloatDef, argument: Reg<F32>) -> Reg<F32> {
        let value = |slot: FloatSlot| Value::from(self.floats[slot]);
        match def {
            FloatDef::Argument => argument,
            FloatDef::Const(bits) => code.mov(PtxType::F32, Value::f32(f32::from_bits(bits))),
            FloatDef::Binary(op, x, y) => code.float(float_op(op), value(x), value(y)),
            FloatDef::Fma(x, y, z) => code.fma(value(x), value(y), value(z)),
            FloatDef::RoundToInteger(x) => {
                code.convert(CvtMode::Rni, (PtxType::F32, PtxType::F32), value(x))
            }
            FloatDef::FromInt(n) => {
                code.convert(CvtMode::Rn, (PtxType::F32, PtxType::S32), self.ints[n].into())
            }
            FloatDef::FromBits(n) => code.reinterpret(PtxType::B32, self.ints[n]),
            FloatDef::Select(p, x, y) => {
                code.select(PtxType::F32, value(x), value(y), self.preds[p])
            }
        }
    }

    fn int(&self, code: &mut FunctionBuilder, def: IntDef) -> Reg<B32> {
        let value = |slot: IntSlot| Value::from(self.ints[slot]);
        match def {
            IntDef::Const(value) => code.mov(PtxType::B32, Value::<B32>::int(value.cast_signed())),
            IntDef::FromFloat(x) => {
                code.convert(CvtMode::Rni, (PtxType::S32, PtxType::F32), self.floats[x].into())
            }
            IntDef::Bits(x) => code.reinterpret(PtxType::B32, self.floats[x]),
            IntDef::Binary(op, x, y) => {
                let (op, signedness) = int_op(op);
                code.int(op, signedness, value(x), value(y))
            }
            IntDef::Select(p, x, y) => code.select(PtxType::B32, value(x), value(y), self.preds[p]),
        }
    }

    fn pred(&self, code: &mut FunctionBuilder, def: PredDef) -> Reg<Pred> {
        match def {
            PredDef::CompareFloat(cmp, x, y) => {
                code.compare_float(compare(cmp), self.floats[x].into(), self.floats[y].into())
            }
            PredDef::CompareInt(cmp, x, y) => code.compare_int(
                compare(cmp),
                Signedness::Signed,
                self.ints[x].into(),
                self.ints[y].into(),
            ),
        }
    }
}

const fn float_op(op: FloatOp) -> BinaryOp {
    match op {
        FloatOp::Add => BinaryOp::Add,
        FloatOp::Sub => BinaryOp::Sub,
        FloatOp::Mul => BinaryOp::Mul,
        FloatOp::Div => BinaryOp::Div,
    }
}

/// The PTX operation and the type it reads its operands as.
const fn int_op(op: IntOp) -> (BinaryOp, Signedness) {
    match op {
        IntOp::Add => (BinaryOp::Add, Signedness::Signed),
        IntOp::Sub => (BinaryOp::Sub, Signedness::Signed),
        IntOp::And => (BinaryOp::And, Signedness::Bits),
        IntOp::Or => (BinaryOp::Or, Signedness::Bits),
        IntOp::Shl => (BinaryOp::Shl, Signedness::Bits),
        IntOp::ShrArithmetic => (BinaryOp::Shr, Signedness::Signed),
        IntOp::ShrLogical => (BinaryOp::Shr, Signedness::Unsigned),
    }
}

const fn compare(cmp: Comparison) -> Compare {
    match cmp {
        Comparison::Lt => Compare::Lt,
        Comparison::Gt => Compare::Gt,
        Comparison::Ge => Compare::Ge,
        Comparison::Eq => Compare::Eq,
    }
}
