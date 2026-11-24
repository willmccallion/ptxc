//! Registers the code sets more than once.

use super::function::FunctionBuilder;
use super::reg::{Class, Reg};

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

impl FunctionBuilder {
    /// A new register to set more than once.
    pub(crate) const fn variable<C: Class>(&mut self) -> Var<C> {
        Var(self.reg())
    }
}
