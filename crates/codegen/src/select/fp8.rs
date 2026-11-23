//! FP8 conversions in hardware, which `sm_89` has from PTX ISA 8.1 (D50): an
//! f32 rounds into the low byte of a packed pair, and a byte widens through
//! f16, which holds every FP8 value exactly.

use crate::ast::{B16, B32, Compare, CvtMode, F32, FunctionBuilder, PtxType, Reg, UnaryOp, Value};
use ptxc_core::float::Overflow;

/// An FP8 format, as ONNX and the OCP specification define it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Fp8 {
    E4m3,
    E5m2,
}

/// The sign bit of an FP8 byte.
const SIGN_BIT: u16 = 0x80;

impl Fp8 {
    /// The packed pair type `cvt` converts.
    const fn pair(self) -> PtxType {
        match self {
            Self::E4m3 => PtxType::E4m3x2,
            Self::E5m2 => PtxType::E5m2x2,
        }
    }

    /// The test a magnitude passes when it rounds beyond the largest
    /// finite value. E4M3's 464 is the tie between 448 and 480, which goes
    /// to 448's even mantissa; E5M2's 61440 is the tie between 57344 and
    /// 65536, which goes to 65536.
    const fn overflows(self) -> (Compare, f32) {
        match self {
            Self::E4m3 => (Compare::Gt, 464.0),
            Self::E5m2 => (Compare::Ge, 61_440.0),
        }
    }

    /// The positive encoding an overflow gives: NaN for E4M3FN, which has
    /// no infinity, and infinity for E5M2.
    const fn non_finite(self) -> u16 {
        match self {
            Self::E4m3 => 0x7F,
            Self::E5m2 => 0x7C,
        }
    }
}

/// `value` rounded to `format`, in the low byte. NaN stays NaN.
pub(super) fn narrow_fp8(
    b: &mut FunctionBuilder,
    value: Reg<F32>,
    format: Fp8,
    overflow: Overflow,
) -> Reg<B16> {
    let saturated = b.pack_fp8(format.pair(), (Value::f32(0.0), value.into()));
    match overflow {
        Overflow::Saturate => saturated,
        Overflow::NonFinite => non_finite_on_overflow(b, value, saturated, format),
    }
}

/// `saturated`, the conversion of `value`, with an overflowed magnitude
/// replaced by the non-finite encoding of `value`'s sign.
fn non_finite_on_overflow(
    b: &mut FunctionBuilder,
    value: Reg<F32>,
    saturated: Reg<B16>,
    format: Fp8,
) -> Reg<B16> {
    let magnitude = b.float_unary(UnaryOp::Abs, value.into());
    let (test, bound) = format.overflows();
    let overflowed = b.compare_float(test, magnitude.into(), Value::f32(bound));
    let negative = b.compare_float(Compare::Lt, value.into(), Value::f32(0.0));
    let positive = format.non_finite();
    let (below, above) = (Value::<B16>::bits(positive | SIGN_BIT), Value::<B16>::bits(positive));
    let special = b.select(PtxType::B16, below, above, negative);
    b.select(PtxType::B16, special.into(), saturated.into(), overflowed)
}

/// The f32 holding the FP8 value in the low byte of `bits`.
pub(super) fn widen_fp8(b: &mut FunctionBuilder, bits: Reg<B16>, format: Fp8) -> Reg<F32> {
    let halves: Reg<B32> = b.convert(CvtMode::Rn, (PtxType::F16x2, format.pair()), bits.into());
    let low: Reg<B16> = b.convert(CvtMode::Plain, (PtxType::U16, PtxType::U32), halves.into());
    b.convert(CvtMode::Plain, (PtxType::F32, PtxType::F16), low.into())
}
