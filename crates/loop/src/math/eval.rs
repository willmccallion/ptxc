//! Evaluates a program on the host, each operation as the PTX codegen
//! emits for it computes: IEEE f32 rounding to nearest even with
//! subnormals kept, wrapping 32-bit integers, ordered comparisons.

use super::program::{
    Comparison, FloatDef, FloatOp, FloatSlot, IntDef, IntOp, IntSlot, MathProgram, PredDef,
    PredSlot, Step,
};
use ptxc_core::IndexVec;

/// The values a program's steps have defined so far.
struct Slots {
    floats: IndexVec<FloatSlot, f32>,
    ints: IndexVec<IntSlot, u32>,
    preds: IndexVec<PredSlot, bool>,
}

impl MathProgram {
    /// The program's result for `x`.
    pub fn eval(&self, x: f32) -> f32 {
        let mut slots =
            Slots { floats: IndexVec::new(), ints: IndexVec::new(), preds: IndexVec::new() };
        for step in &self.steps {
            match *step {
                Step::Float(def) => {
                    let value = slots.float(def, x);
                    let _slot = slots.floats.push(value);
                }
                Step::Int(def) => {
                    let value = slots.int(def);
                    let _slot = slots.ints.push(value);
                }
                Step::Pred(def) => {
                    let value = slots.pred(def);
                    let _slot = slots.preds.push(value);
                }
            }
        }
        slots.floats[self.result]
    }
}

impl Slots {
    fn float(&self, def: FloatDef, argument: f32) -> f32 {
        let f = |slot: FloatSlot| self.floats[slot];
        match def {
            FloatDef::Argument => argument,
            FloatDef::Const(bits) => f32::from_bits(bits),
            FloatDef::Binary(op, a, b) => float_binary(op, f(a), f(b)),
            FloatDef::Fma(a, b, c) => f(a).mul_add(f(b), f(c)),
            FloatDef::RoundToInteger(a) => f(a).round_ties_even(),
            FloatDef::FromInt(n) => int_to_float(self.ints[n]),
            FloatDef::FromBits(bits) => f32::from_bits(self.ints[bits]),
            FloatDef::Select(p, a, b) => {
                if self.preds[p] {
                    f(a)
                } else {
                    f(b)
                }
            }
        }
    }

    fn int(&self, def: IntDef) -> u32 {
        let n = |slot: IntSlot| self.ints[slot];
        match def {
            IntDef::Const(value) => value,
            IntDef::FromFloat(a) => float_to_int(self.floats[a]),
            IntDef::Bits(a) => self.floats[a].to_bits(),
            IntDef::Binary(op, a, b) => int_binary(op, n(a), n(b)),
            IntDef::Select(p, a, b) => {
                if self.preds[p] {
                    n(a)
                } else {
                    n(b)
                }
            }
        }
    }

    fn pred(&self, def: PredDef) -> bool {
        match def {
            PredDef::CompareFloat(cmp, a, b) => compare(cmp, self.floats[a], self.floats[b]),
            PredDef::CompareInt(cmp, a, b) => {
                compare(cmp, self.ints[a].cast_signed(), self.ints[b].cast_signed())
            }
        }
    }
}

fn float_binary(op: FloatOp, a: f32, b: f32) -> f32 {
    match op {
        FloatOp::Add => a + b,
        FloatOp::Sub => a - b,
        FloatOp::Mul => a * b,
        FloatOp::Div => a / b,
    }
}

const fn int_binary(op: IntOp, a: u32, b: u32) -> u32 {
    match op {
        IntOp::Add => a.wrapping_add(b),
        IntOp::Sub => a.wrapping_sub(b),
        IntOp::And => a & b,
        IntOp::Or => a | b,
        IntOp::Shl => a << b,
        IntOp::ShrArithmetic => (a.cast_signed() >> b).cast_unsigned(),
        IntOp::ShrLogical => a >> b,
    }
}

/// `n` as two's complement, rounded to the nearest f32, as
/// `cvt.rn.f32.s32` does.
#[expect(clippy::cast_precision_loss, reason = "rounds to nearest even, as cvt.rn does")]
const fn int_to_float(n: u32) -> f32 {
    n.cast_signed() as f32
}

/// An integral `x` as two's complement, saturating with NaN giving 0, as
/// `cvt.rni.s32.f32` does.
#[expect(clippy::cast_possible_truncation, reason = "saturates, as cvt.rni does")]
const fn float_to_int(x: f32) -> u32 {
    (x.round_ties_even() as i32).cast_unsigned()
}

fn compare<T: PartialOrd + Copy>(cmp: Comparison, a: T, b: T) -> bool {
    match cmp {
        Comparison::Lt => a < b,
        Comparison::Gt => a > b,
        Comparison::Ge => a >= b,
        Comparison::Eq => a == b,
    }
}
