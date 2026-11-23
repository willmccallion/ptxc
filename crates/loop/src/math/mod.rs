//! Math programs (D51): the transcendental functions as straight-line code.
//!
//! Every operation is exact IEEE f32 or 32-bit integer arithmetic. The loop
//! interpreter evaluates a program and codegen emits it as PTX, so the two
//! agree bit for bit, NaN payloads aside.

mod builder;
mod constants;
mod erf;
mod eval;
mod exp;
mod expm1;
mod log;
mod log1p;
mod program;
mod reduce;
mod sincos;
mod tanh;

#[cfg(test)]
mod semantics;

pub use program::{
    Comparison, FloatDef, FloatOp, FloatSlot, IntDef, IntOp, IntSlot, MathProgram, PredDef,
    PredSlot, Step,
};

use std::sync::LazyLock;

static EXP: LazyLock<MathProgram> = LazyLock::new(exp::exp);
static LOG: LazyLock<MathProgram> = LazyLock::new(log::log);
static EXPM1: LazyLock<MathProgram> = LazyLock::new(expm1::expm1);
static LOG1P: LazyLock<MathProgram> = LazyLock::new(log1p::log1p);
static TANH: LazyLock<MathProgram> = LazyLock::new(tanh::tanh);
static ERF: LazyLock<MathProgram> = LazyLock::new(erf::erf);
static SIN: LazyLock<MathProgram> = LazyLock::new(sincos::sin);
static COS: LazyLock<MathProgram> = LazyLock::new(sincos::cos);

/// `e^x`, within 1.06 ulps.
pub fn exp() -> &'static MathProgram {
    &EXP
}

/// The natural logarithm, within 0.84 ulps.
pub fn log() -> &'static MathProgram {
    &LOG
}

/// `e^x - 1`, within 1.65 ulps.
pub fn expm1() -> &'static MathProgram {
    &EXPM1
}

/// `log(1 + x)`, within 1.44 ulps.
pub fn log1p() -> &'static MathProgram {
    &LOG1P
}

/// The hyperbolic tangent, within 2.42 ulps.
pub fn tanh() -> &'static MathProgram {
    &TANH
}

/// The error function, within 1.03 ulps.
pub fn erf() -> &'static MathProgram {
    &ERF
}

/// The sine, within 2.25 ulps for every finite f32.
pub fn sin() -> &'static MathProgram {
    &SIN
}

/// The cosine, within 2.26 ulps for every finite f32.
pub fn cos() -> &'static MathProgram {
    &COS
}
