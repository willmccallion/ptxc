//! Instruction operands: registers, immediates and addresses.

use super::arithmetic::Int;
use super::reg::{B16, B32, B64, Class, F32, Reg, RegId};

/// A source operand as instructions store it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Operand {
    Reg(RegId),
    /// An integer immediate.
    Int(i64),
    /// An f32 immediate, by its bits, printed exactly.
    F32(u32),
}

/// A source operand of class `C`: a register or an immediate of its kind.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct Value<C> {
    operand: Operand,
    class: std::marker::PhantomData<C>,
}

impl<C: Class> Value<C> {
    pub(crate) const fn operand(self) -> Operand {
        self.operand
    }
}

impl<C: Class> From<Reg<C>> for Value<C> {
    fn from(reg: Reg<C>) -> Self {
        Self { operand: Operand::Reg(reg.id()), class: std::marker::PhantomData }
    }
}

impl<C: Int> Value<C> {
    /// An integer immediate of class `C`, which must fit it.
    pub(crate) const fn immediate(value: i64) -> Self {
        Self { operand: Operand::Int(value), class: std::marker::PhantomData }
    }
}

impl Value<F32> {
    pub(crate) const fn f32(value: f32) -> Self {
        Self { operand: Operand::F32(value.to_bits()), class: std::marker::PhantomData }
    }
}

impl Value<B16> {
    pub(crate) fn bits(value: u16) -> Self {
        Self { operand: Operand::Int(i64::from(value)), class: std::marker::PhantomData }
    }
}

impl Value<B32> {
    pub(crate) fn int(value: i32) -> Self {
        Self { operand: Operand::Int(i64::from(value)), class: std::marker::PhantomData }
    }
}

impl Value<B64> {
    pub(crate) const fn int(value: i64) -> Self {
        Self { operand: Operand::Int(value), class: std::marker::PhantomData }
    }
}

/// Where a load or store goes: a register holding an address, or a named
/// array, plus a byte offset.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Address {
    Reg { base: RegId, offset: i64 },
    Array { name: String, offset: i64 },
}

/// A special register a thread reads its position from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Special {
    ThreadX,
    ThreadY,
    ThreadZ,
    BlockX,
    BlockY,
    BlockZ,
    /// The thread's lane within its warp.
    LaneId,
}

impl Special {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::ThreadX => "%tid.x",
            Self::ThreadY => "%tid.y",
            Self::ThreadZ => "%tid.z",
            Self::BlockX => "%ctaid.x",
            Self::BlockY => "%ctaid.y",
            Self::BlockZ => "%ctaid.z",
            Self::LaneId => "%laneid",
        }
    }
}
