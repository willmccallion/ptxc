//! Selects casts between every pair of element types: floats round,
//! floats into integers truncate and saturate as `cvt.rzi.sat`, integers
//! wrap, and booleans are 0 and 1.

use super::Selector;
use super::values::{Held, wrap32};
use crate::ast::{
    B32, B64, BinaryOp, Class, Compare, CvtMode, F32, FunctionBuilder, Pred, PtxType, Reg,
    Signedness, Value,
};
use ptxc_core::DType;
use ptxc_core::float::Overflow;

/// The type an integer held in 32 bits is read as.
const fn held_type(dtype: DType) -> PtxType {
    if matches!(dtype, DType::U8) { PtxType::U32 } else { PtxType::S32 }
}

impl Selector<'_> {
    /// `value`, of type `from`, converted to `to`; FP8 overflow follows
    /// `overflow`.
    pub(super) fn cast(
        &mut self,
        value: Held,
        (from, to): (DType, DType),
        overflow: Overflow,
    ) -> Held {
        match value {
            Held::Bool(flag) => self.cast_bool(flag, to),
            Held::Float(x) => self.cast_float(x, to, overflow),
            Held::Int32(x) => self.cast_int32(x, (from, to), overflow),
            Held::Int64(x) => self.cast_int64(x, to, overflow),
        }
    }

    fn cast_bool(&mut self, flag: Reg<Pred>, to: DType) -> Held {
        let b = &mut self.builder;
        match to {
            DType::Bool => Held::Bool(flag),
            DType::I64 => Held::Int64(b.select(
                PtxType::B64,
                Value::<B64>::int(1),
                Value::<B64>::int(0),
                flag,
            )),
            _ if to.is_float() => {
                Held::Float(b.select(PtxType::F32, Value::f32(1.0), Value::f32(0.0), flag))
            }
            _ => Held::Int32(b.select(
                PtxType::B32,
                Value::<B32>::int(1),
                Value::<B32>::int(0),
                flag,
            )),
        }
    }

    fn cast_float(&mut self, x: Reg<F32>, to: DType, overflow: Overflow) -> Held {
        match to {
            DType::I8 => Held::Int32(self.truncated(x, PtxType::S8)),
            DType::U8 => Held::Int32(self.truncated(x, PtxType::U8)),
            DType::I32 => Held::Int32(self.truncated(x, PtxType::S32)),
            DType::I64 => Held::Int64(self.truncated(x, PtxType::S64)),
            DType::Bool => {
                let zero = Value::f32(0.0);
                Held::Bool(self.builder.compare_float(Compare::NeOrUnordered, x.into(), zero))
            }
            DType::F32 | DType::F16 | DType::Bf16 | DType::F8E4M3Fn | DType::F8E5M2 => {
                Held::Float(self.rounded(x, to, overflow))
            }
        }
    }

    /// `x` truncated toward zero into the integer type `ty`, saturating;
    /// NaN gives 0.
    fn truncated<C: Class>(&mut self, x: Reg<F32>, ty: PtxType) -> Reg<C> {
        self.builder.convert(CvtMode::RziSat, (ty, PtxType::F32), x.into())
    }

    fn cast_int32(&mut self, x: Reg<B32>, (from, to): (DType, DType), overflow: Overflow) -> Held {
        let b = &mut self.builder;
        let source = held_type(from);
        match to {
            DType::Bool => {
                let zero = Value::<B32>::int(0);
                Held::Bool(b.compare_int(Compare::Ne, Signedness::Signed, x.into(), zero))
            }
            DType::I64 => Held::Int64(b.convert(CvtMode::Plain, (PtxType::S64, source), x.into())),
            DType::F32 => Held::Float(b.convert(CvtMode::Rn, (PtxType::F32, source), x.into())),
            _ if to.is_float() => {
                let wide = b.convert(CvtMode::Plain, (PtxType::S64, source), x.into());
                Held::Float(self.narrow_integer(wide, to, overflow))
            }
            _ => Held::Int32(wrap32(b, x, to)),
        }
    }

    fn cast_int64(&mut self, x: Reg<B64>, to: DType, overflow: Overflow) -> Held {
        let b = &mut self.builder;
        match to {
            DType::Bool => {
                let zero = Value::<B64>::int(0);
                Held::Bool(b.compare_int(Compare::Ne, Signedness::Signed, x.into(), zero))
            }
            DType::I64 => Held::Int64(x),
            DType::F32 => {
                Held::Float(b.convert(CvtMode::Rn, (PtxType::F32, PtxType::S64), x.into()))
            }
            _ if to.is_float() => Held::Float(self.narrow_integer(x, to, overflow)),
            _ => {
                let low: Reg<B32> =
                    b.convert(CvtMode::Plain, (PtxType::U32, PtxType::U64), x.into());
                Held::Int32(wrap32(b, low, to))
            }
        }
    }

    /// Integer `x` rounded once to `to`, a float type narrower than f32.
    /// `sm_89` converts integers only to f32 (`cvt.rn.bf16.s32` needs
    /// `sm_90`), so the f32 is rounded to odd, whose spare bits make the
    /// second rounding give the correctly rounded result.
    fn narrow_integer(&mut self, x: Reg<B64>, to: DType, overflow: Overflow) -> Reg<F32> {
        let odd = rounded_to_odd(&mut self.builder, x);
        self.rounded(odd, to, overflow)
    }
}

/// `x` as an f32 rounded to odd: truncated toward zero, with the last
/// mantissa bit set when truncating lost bits.
fn rounded_to_odd(b: &mut FunctionBuilder, x: Reg<B64>) -> Reg<F32> {
    let truncated: Reg<F32> = b.convert(CvtMode::Rz, (PtxType::F32, PtxType::S64), x.into());
    let back: Reg<B64> = b.convert(CvtMode::RziSat, (PtxType::S64, PtxType::F32), truncated.into());
    let inexact = b.compare_int(Compare::Ne, Signedness::Signed, back.into(), x.into());
    let sticky = b.select(PtxType::B32, Value::<B32>::int(1), Value::<B32>::int(0), inexact);
    let bits: Reg<B32> = b.reinterpret(PtxType::B32, truncated);
    let odd = b.int(BinaryOp::Or, Signedness::Bits, bits.into(), sticky.into());
    b.reinterpret(PtxType::B32, odd)
}
