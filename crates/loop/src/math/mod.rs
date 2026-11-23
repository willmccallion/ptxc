//! Math programs (D51): the transcendental functions as straight-line code.
//!
//! Every operation is exact IEEE f32 or 32-bit integer arithmetic. The loop
//! interpreter evaluates a program and codegen emits it as PTX, so the two
//! agree bit for bit, NaN payloads aside.

mod builder;
mod constants;
mod eval;
mod exp;
mod log;
mod program;

#[cfg(test)]
mod semantics;

pub use program::{
    Comparison, FloatDef, FloatOp, FloatSlot, IntDef, IntOp, IntSlot, MathProgram, PredDef,
    PredSlot, Step,
};

use std::sync::LazyLock;

static EXP: LazyLock<MathProgram> = LazyLock::new(exp::exp);
static LOG: LazyLock<MathProgram> = LazyLock::new(log::log);

/// `e^x`, within 1.06 ulps.
pub fn exp() -> &'static MathProgram {
    &EXP
}

/// The natural logarithm, within 0.84 ulps.
pub fn log() -> &'static MathProgram {
    &LOG
}
