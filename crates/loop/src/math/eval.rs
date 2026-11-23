//! Evaluates a program on the host, each operation as the PTX codegen
//! emits for it computes: IEEE f32 rounding to nearest even with
//! subnormals kept, wrapping 32-bit integers, ordered comparisons.

use super::program::{
    Binary, Comparison, FloatDef, FloatOp, FloatSlot, IntDef, IntOp, IntSlot, MathProgram, PredDef,
    PredSlot, Side, Step, Unary,
};
use ptxc_core::IndexVec;

/// The values a program's steps have defined so far.
struct Slots {
    floats: IndexVec<FloatSlot, f32>,
    ints: IndexVec<IntSlot, u32>,
    preds: IndexVec<PredSlot, bool>,
}

impl MathProgram<Unary> {
    /// The program's result for `x`.
    pub fn eval(&self, x: f32) -> f32 {
        run(&self.steps, self.result, [x, x])
    }
}

impl MathProgram<Binary> {
    /// The program's result for `x` and `y`.
    pub fn eval(&self, x: f32, y: f32) -> f32 {
        run(&self.steps, self.result, [x, y])
    }
}

/// `steps` run on `arguments`, left then right; a unary program reads only
/// the left.
fn run(steps: &[Step], result: FloatSlot, arguments: [f32; 2]) -> f32 {
    let mut slots =
        Slots { floats: IndexVec::new(), ints: IndexVec::new(), preds: IndexVec::new() };
    for step in steps {
        match *step {
            Step::Float(def) => {
                let value = slots.float(def, arguments);
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
    slots.floats[result]
}

impl Slots {
    fn float(&self, def: FloatDef, [left, right]: [f32; 2]) -> f32 {
        let f = |slot: FloatSlot| self.floats[slot];
        match def {
            FloatDef::Argument(Side::Left) => left,
            FloatDef::Argument(Side::Right) => right,
            FloatDef::Const(bits) => f32::from_bits(bits),
            FloatDef::Binary(op, a, b) => float_binary(op, f(a), f(b)),
            FloatDef::Fma(a, b, c) => f(a).mul_add(f(b), f(c)),
            FloatDef::RoundToInteger(a) => f(a).round_ties_even(),
            FloatDef::FromInt(n) => int_to_float(self.ints[n]),
            FloatDef::FromUnsigned(n) => unsigned_to_float(self.ints[n]),
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
            IntDef::LeadingZeros(a) => n(a).leading_zeros(),
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
            PredDef::CompareUnsigned(cmp, a, b) => compare(cmp, self.ints[a], self.ints[b]),
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

fn int_binary(op: IntOp, a: u32, b: u32) -> u32 {
    match op {
        IntOp::Add => a.wrapping_add(b),
        IntOp::Sub => a.wrapping_sub(b),
        IntOp::Mul => a.wrapping_mul(b),
        IntOp::MulHiUnsigned => high_half(a, b),
        IntOp::And => a & b,
        IntOp::Or => a | b,
        IntOp::Xor => a ^ b,
        IntOp::Shl => {
            if b < u32::BITS {
                a << b
            } else {
                0
            }
        }
        IntOp::ShrArithmetic => (a.cast_signed() >> b.min(u32::BITS - 1)).cast_unsigned(),
        IntOp::ShrLogical => {
            if b < u32::BITS {
                a >> b
            } else {
                0
            }
        }
    }
}

/// `n` as two's complement, rounded to the nearest f32, as
/// `cvt.rn.f32.s32` does.
#[expect(clippy::cast_precision_loss, reason = "rounds to nearest even, as cvt.rn does")]
const fn int_to_float(n: u32) -> f32 {
    n.cast_signed() as f32
}

/// The high 32 bits of `a * b`, as `mul.hi.u32` gives them.
fn high_half(a: u32, b: u32) -> u32 {
    let [_, _, _, _, high @ ..] = (u64::from(a) * u64::from(b)).to_le_bytes();
    u32::from_le_bytes(high)
}

/// `n` rounded to the nearest f32, as `cvt.rn.f32.u32` does.
#[expect(clippy::cast_precision_loss, reason = "rounds to nearest even, as cvt.rn does")]
const fn unsigned_to_float(n: u32) -> f32 {
    n as f32
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
