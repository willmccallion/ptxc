//! How a loop IR value of each element type is held in registers, and its
//! form in memory: floats in f32 registers, narrower ones exactly; 8- and
//! 32-bit integers in 32-bit registers; `i64` in 64-bit; bools as
//! predicates.

use super::fp8::{Fp8, narrow_fp8, widen_fp8};
use crate::ast::{
    B16, B32, B64, BinaryOp, CvtMode, F32, FunctionBuilder, Pred, PtxType, Reg, Signedness, Value,
};
use ptxc_core::DType;
use ptxc_core::float::Overflow;

/// A value in registers.
#[derive(Clone, Copy, Debug)]
pub(super) enum Held {
    Float(Reg<F32>),
    Int32(Reg<B32>),
    Int64(Reg<B64>),
    Bool(Reg<Pred>),
}

/// A float type narrower than f32.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Narrow {
    F16,
    Bf16,
    Fp8(Fp8),
}

impl Narrow {
    /// `dtype`'s kind if it is a float narrower than f32.
    pub(super) const fn of(dtype: DType) -> Option<Self> {
        match dtype {
            DType::F16 => Some(Self::F16),
            DType::Bf16 => Some(Self::Bf16),
            DType::F8E4M3Fn => Some(Self::Fp8(Fp8::E4m3)),
            DType::F8E5M2 => Some(Self::Fp8(Fp8::E5m2)),
            DType::F32 | DType::I8 | DType::U8 | DType::I32 | DType::I64 | DType::Bool => None,
        }
    }
}

/// An element in memory form, about to be stored.
#[derive(Clone, Copy, Debug)]
pub(super) enum Stored {
    F32(Reg<F32>),
    /// f16 or bf16, or an FP8 byte in the low half.
    Half(Reg<B16>),
    Int32(Reg<B32>),
    Int64(Reg<B64>),
}

/// The type a load or store of `dtype` uses.
pub(super) const fn memory_type(dtype: DType) -> PtxType {
    match dtype {
        DType::F32 => PtxType::F32,
        DType::F16 | DType::Bf16 => PtxType::B16,
        DType::F8E4M3Fn | DType::F8E5M2 => PtxType::B8,
        DType::I8 => PtxType::S8,
        DType::U8 | DType::Bool => PtxType::U8,
        DType::I32 => PtxType::S32,
        DType::I64 => PtxType::S64,
    }
}

/// An f32 rounded into the memory form of `kind`.
pub(super) fn narrow(
    b: &mut FunctionBuilder,
    value: Reg<F32>,
    kind: Narrow,
    overflow: Overflow,
) -> Reg<B16> {
    match kind {
        Narrow::F16 => b.convert(CvtMode::Rn, (PtxType::F16, PtxType::F32), value.into()),
        Narrow::Bf16 => b.convert(CvtMode::Rn, (PtxType::Bf16, PtxType::F32), value.into()),
        Narrow::Fp8(format) => narrow_fp8(b, value, format, overflow),
    }
}

/// The f32 holding a value in the memory form of `kind`.
pub(super) fn widen(b: &mut FunctionBuilder, value: Reg<B16>, kind: Narrow) -> Reg<F32> {
    match kind {
        Narrow::F16 => b.convert(CvtMode::Plain, (PtxType::F32, PtxType::F16), value.into()),
        Narrow::Bf16 => {
            let bits: Reg<B32> =
                b.convert(CvtMode::Plain, (PtxType::U32, PtxType::U16), value.into());
            let shifted =
                b.int(BinaryOp::Shl, Signedness::Bits, bits.into(), Value::<B32>::int(16));
            b.reinterpret(PtxType::B32, shifted)
        }
        Narrow::Fp8(format) => widen_fp8(b, value, format),
    }
}

/// A held value of `dtype` in memory form, for a store. Narrow floats
/// overflow to their non-finite values, as every rounding but a saturating
/// cast does.
pub(super) fn stored(b: &mut FunctionBuilder, value: Held, dtype: DType) -> Stored {
    match value {
        Held::Float(value) => Narrow::of(dtype).map_or(Stored::F32(value), |kind| {
            Stored::Half(narrow(b, value, kind, Overflow::NonFinite))
        }),
        Held::Int32(value) => Stored::Int32(value),
        Held::Int64(value) => Stored::Int64(value),
        Held::Bool(value) => {
            Stored::Int32(b.select(PtxType::B32, Value::<B32>::int(1), Value::<B32>::int(0), value))
        }
    }
}

/// An integer `value` of `dtype`, computed in 32 bits, wrapped to it: 8-bit
/// types sign- or zero-extend their low byte.
pub(super) fn wrap32(b: &mut FunctionBuilder, value: Reg<B32>, dtype: DType) -> Reg<B32> {
    match dtype {
        DType::I8 => b.convert(CvtMode::Plain, (PtxType::S32, PtxType::S8), value.into()),
        DType::U8 => b.convert(CvtMode::Plain, (PtxType::U32, PtxType::U8), value.into()),
        _ => value,
    }
}
