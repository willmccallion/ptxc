//! Which instruction defines each register, so an analysis can follow a
//! value back through the instructions that computed it. Only a `Var` is
//! set more than once, and it is known to be from its start, so a register
//! that has one definition so far keeps it.

use super::function::FunctionBuilder;
use super::instr::Instr;
use super::reg::{Class, Reg, RegId};
use std::slice;

/// Where a register gets its value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Definition {
    /// From the body's instruction at this position, its only definition.
    Once(usize),
    /// From several instructions: a `Var`'s.
    Several,
}

/// A register the code sets more than once, such as a loop's counter or a
/// word of an array held in registers: the only kind `assign` and
/// `increment` write.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Var<C>(Reg<C>);

impl<C: Class> Var<C> {
    pub(crate) const fn reg(self) -> Reg<C> {
        self.0
    }
}

impl Instr {
    /// The registers the instruction writes.
    pub(crate) fn destinations(&self) -> &[RegId] {
        match self {
            Self::Mov { dst, .. }
            | Self::Special { dst, .. }
            | Self::Binary { dst, .. }
            | Self::Fma { dst, .. }
            | Self::Unary { dst, .. }
            | Self::Setp { dst, .. }
            | Self::Selp { dst, .. }
            | Self::Cvt { dst, .. }
            | Self::Ld { dst, .. }
            | Self::ToGlobal { dst, .. }
            | Self::AddressOf { dst, .. }
            | Self::Increment { dst } => slice::from_ref(dst),
            Self::LdVector { dst, .. } | Self::LdMatrix { dst, .. } => dst,
            Self::Mma { d, .. } => d,
            Self::St { .. }
            | Self::Label(_)
            | Self::Bra { .. }
            | Self::BarSync
            | Self::StVector { .. }
            | Self::CpAsync { .. }
            | Self::CpAsyncCommit
            | Self::CpAsyncWait { .. }
            | Self::Ret => &[],
        }
    }
}

impl FunctionBuilder {
    /// A new register to set more than once.
    pub(crate) fn variable<C: Class>(&mut self) -> Var<C> {
        let reg = self.reg();
        let _new = self.definitions.insert(reg.id(), Definition::Several);
        Var(reg)
    }

    /// The instruction that alone defines `reg`, if one does.
    pub(crate) fn definition(&self, reg: RegId) -> Option<&Instr> {
        match self.definitions.get(&reg)? {
            Definition::Once(position) => self.function.body.get(*position),
            Definition::Several => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{B64, FunctionBuilder, Instr, PtxType, Reg, Value};

    #[test]
    fn a_registers_one_definition_is_the_instruction_that_set_it() {
        let mut builder = FunctionBuilder::new("kernel");

        let set: Reg<B64> = builder.mov(PtxType::S64, Value::<B64>::int(3));

        let found = builder.definition(set.id());
        assert!(matches!(found, Some(Instr::Mov { .. })), "{found:?}");
    }

    #[test]
    fn a_variable_has_no_one_definition_even_before_it_is_set_again() {
        let mut builder = FunctionBuilder::new("kernel");
        let counter = builder.variable::<B64>();

        builder.assign(counter, PtxType::S64, Value::<B64>::int(0));

        assert_eq!(builder.definition(counter.reg().id()), None);
    }
}
