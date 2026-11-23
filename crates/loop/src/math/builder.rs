//! `ProgramBuilder`: writes a program a step at a time, numbering each
//! kind's slots in the order the steps define them.

use super::program::{
    Comparison, FloatDef, FloatOp, FloatSlot, IntDef, IntOp, IntSlot, MathProgram, PredDef,
    PredSlot, Step,
};
use ptxc_core::Idx;

pub(super) struct ProgramBuilder {
    steps: Vec<Step>,
    floats: usize,
    ints: usize,
    preds: usize,
}

impl ProgramBuilder {
    /// A builder whose first float is the argument.
    pub(super) fn new() -> (Self, FloatSlot) {
        let mut builder = Self { steps: Vec::new(), floats: 0, ints: 0, preds: 0 };
        let argument = builder.define_float(FloatDef::Argument);
        (builder, argument)
    }

    pub(super) fn finish(self, result: FloatSlot) -> MathProgram {
        MathProgram { steps: self.steps, result }
    }

    fn define_float(&mut self, def: FloatDef) -> FloatSlot {
        self.steps.push(Step::Float(def));
        self.floats += 1;
        FloatSlot::new(self.floats - 1)
    }

    fn define_int(&mut self, def: IntDef) -> IntSlot {
        self.steps.push(Step::Int(def));
        self.ints += 1;
        IntSlot::new(self.ints - 1)
    }

    fn define_pred(&mut self, def: PredDef) -> PredSlot {
        self.steps.push(Step::Pred(def));
        self.preds += 1;
        PredSlot::new(self.preds - 1)
    }

    pub(super) fn float(&mut self, value: f32) -> FloatSlot {
        self.define_float(FloatDef::Const(value.to_bits()))
    }

    pub(super) fn add(&mut self, a: FloatSlot, b: FloatSlot) -> FloatSlot {
        self.define_float(FloatDef::Binary(FloatOp::Add, a, b))
    }

    pub(super) fn sub(&mut self, a: FloatSlot, b: FloatSlot) -> FloatSlot {
        self.define_float(FloatDef::Binary(FloatOp::Sub, a, b))
    }

    pub(super) fn mul(&mut self, a: FloatSlot, b: FloatSlot) -> FloatSlot {
        self.define_float(FloatDef::Binary(FloatOp::Mul, a, b))
    }

    pub(super) fn div(&mut self, a: FloatSlot, b: FloatSlot) -> FloatSlot {
        self.define_float(FloatDef::Binary(FloatOp::Div, a, b))
    }

    /// `a * b + c`, rounded once.
    pub(super) fn fma(&mut self, a: FloatSlot, b: FloatSlot, c: FloatSlot) -> FloatSlot {
        self.define_float(FloatDef::Fma(a, b, c))
    }

    /// `lowest + x * (higher[0] + x * (higher[1] + ...))`, by Horner's rule
    /// with fused steps.
    pub(super) fn polynomial(&mut self, x: FloatSlot, lowest: f32, higher: &[f32]) -> FloatSlot {
        let Some((&highest, middle)) = higher.split_last() else {
            return self.float(lowest);
        };
        let mut sum = self.float(highest);
        for &coefficient in middle.iter().rev().chain(std::iter::once(&lowest)) {
            let constant = self.float(coefficient);
            sum = self.fma(sum, x, constant);
        }
        sum
    }

    pub(super) fn round_to_integer(&mut self, x: FloatSlot) -> FloatSlot {
        self.define_float(FloatDef::RoundToInteger(x))
    }

    pub(super) fn int_to_float(&mut self, n: IntSlot) -> FloatSlot {
        self.define_float(FloatDef::FromInt(n))
    }

    pub(super) fn float_from_bits(&mut self, bits: IntSlot) -> FloatSlot {
        self.define_float(FloatDef::FromBits(bits))
    }

    pub(super) fn select(&mut self, p: PredSlot, a: FloatSlot, b: FloatSlot) -> FloatSlot {
        self.define_float(FloatDef::Select(p, a, b))
    }

    pub(super) fn int(&mut self, value: i32) -> IntSlot {
        self.define_int(IntDef::Const(value.cast_unsigned()))
    }

    pub(super) fn float_to_int(&mut self, x: FloatSlot) -> IntSlot {
        self.define_int(IntDef::FromFloat(x))
    }

    pub(super) fn bits(&mut self, x: FloatSlot) -> IntSlot {
        self.define_int(IntDef::Bits(x))
    }

    pub(super) fn int_op(&mut self, op: IntOp, a: IntSlot, b: IntSlot) -> IntSlot {
        self.define_int(IntDef::Binary(op, a, b))
    }

    pub(super) fn select_int(&mut self, p: PredSlot, a: IntSlot, b: IntSlot) -> IntSlot {
        self.define_int(IntDef::Select(p, a, b))
    }

    pub(super) fn compare(&mut self, cmp: Comparison, a: FloatSlot, b: FloatSlot) -> PredSlot {
        self.define_pred(PredDef::CompareFloat(cmp, a, b))
    }

    pub(super) fn compare_int(&mut self, cmp: Comparison, a: IntSlot, b: IntSlot) -> PredSlot {
        self.define_pred(PredDef::CompareInt(cmp, a, b))
    }
}
