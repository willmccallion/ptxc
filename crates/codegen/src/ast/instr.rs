//! `Instr`: the PTX instructions ptxc emits, stored with their types and
//! erased registers. Only `FunctionBuilder`'s typed methods make them.

use super::operand::{Address, Operand, Special};
use super::reg::RegId;

/// An instruction's type suffix.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum PtxType {
    Pred,
    B8,
    B16,
    B32,
    B64,
    U8,
    U16,
    U32,
    U64,
    S8,
    S32,
    S64,
    F16,
    Bf16,
    F32,
    /// Two f16 values in 32 bits.
    F16x2,
    /// Two FP8 values in 16 bits, the first in the high byte.
    E4m3x2,
    E5m2x2,
}

impl PtxType {
    pub(crate) const fn suffix(self) -> &'static str {
        match self {
            Self::Pred => ".pred",
            Self::B8 => ".b8",
            Self::B16 => ".b16",
            Self::B32 => ".b32",
            Self::B64 => ".b64",
            Self::U8 => ".u8",
            Self::U16 => ".u16",
            Self::U32 => ".u32",
            Self::U64 => ".u64",
            Self::S8 => ".s8",
            Self::S32 => ".s32",
            Self::S64 => ".s64",
            Self::F16 => ".f16",
            Self::Bf16 => ".bf16",
            Self::F32 => ".f32",
            Self::F16x2 => ".f16x2",
            Self::E4m3x2 => ".e4m3x2",
            Self::E5m2x2 => ".e5m2x2",
        }
    }
}

/// A two-operand operation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    /// `mul.lo` on integers, `mul.rn` on floats.
    Mul,
    Div,
    Rem,
    /// NaN-propagating on floats (`min.NaN`).
    Min,
    /// NaN-propagating on floats (`max.NaN`).
    Max,
    And,
    Or,
    Xor,
    Shl,
    /// Arithmetic on signed types, logical on unsigned and bit types.
    Shr,
    /// The high half of the product.
    MulHi,
}

/// A one-operand operation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum UnaryOp {
    /// Leading zero bits.
    Clz,
    Neg,
    Abs,
    Not,
    Sqrt,
    Rcp,
}

/// A comparison `setp` makes; on floats, ordered except `Unordered`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Compare {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// Whether either operand is NaN.
    Unordered,
    /// Not equal, or either operand NaN.
    NeOrUnordered,
}

/// How a conversion rounds or saturates.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CvtMode {
    /// Integer widening or narrowing by bits, or an exact float widening.
    Plain,
    /// Round to nearest even, into a float type.
    Rn,
    /// Round toward zero, into a float type.
    Rz,
    /// Round to nearest even, saturating to the finite range (FP8).
    RnSatfinite,
    /// Truncate toward zero and saturate, float into an integer type.
    RziSat,
    /// Round a float to an integral float: nearest even, toward negative
    /// or toward positive infinity.
    Rni,
    Rmi,
    Rpi,
}

/// The memory space a load or store addresses.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Space {
    Global,
    Shared,
    Local,
    Param,
}

/// A branch label within a function.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Label(pub(crate) u32);

/// One instruction.
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum Instr {
    Mov {
        ty: PtxType,
        dst: RegId,
        src: Operand,
    },
    Special {
        dst: RegId,
        special: Special,
    },
    Binary {
        op: BinaryOp,
        ty: PtxType,
        dst: RegId,
        a: Operand,
        b: Operand,
    },
    /// `fma.rn.f32`: `a * b + c`, rounded once.
    Fma {
        dst: RegId,
        a: Operand,
        b: Operand,
        c: Operand,
    },
    Unary {
        op: UnaryOp,
        ty: PtxType,
        dst: RegId,
        a: Operand,
    },
    Setp {
        cmp: Compare,
        ty: PtxType,
        dst: RegId,
        a: Operand,
        b: Operand,
    },
    Selp {
        ty: PtxType,
        dst: RegId,
        a: Operand,
        b: Operand,
        pred: RegId,
    },
    Cvt {
        mode: CvtMode,
        to: PtxType,
        from: PtxType,
        dst: RegId,
        src: Vec<Operand>,
    },
    Ld {
        space: Space,
        ty: PtxType,
        dst: RegId,
        addr: Address,
    },
    St {
        space: Space,
        ty: PtxType,
        addr: Address,
        src: Operand,
    },
    /// A generic pointer as a global address.
    ToGlobal {
        dst: RegId,
        src: RegId,
    },
    /// The address of an array.
    AddressOf {
        dst: RegId,
        name: String,
    },
    /// `dst = dst + 1`, for loop counters.
    Increment {
        dst: RegId,
    },
    Label(Label),
    Bra {
        target: Label,
        guard: Option<(RegId, bool)>,
    },
    BarSync,
    Ret,
}
