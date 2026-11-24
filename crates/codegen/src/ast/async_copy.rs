//! `FunctionBuilder`'s asynchronous copies: `cp.async` from global to
//! shared memory, and the commits and waits that group and complete them.

use super::function::FunctionBuilder;
use super::instr::Instr;
use super::operand::Address;
use super::reg::{B32, Reg};

/// The bytes one `cp.async` copies.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CopySize {
    Four,
    Eight,
    Sixteen,
}

impl CopySize {
    /// The size of a copy of `bytes`, if `cp.async` makes one.
    pub(crate) const fn of(bytes: u64) -> Option<Self> {
        match bytes {
            4 => Some(Self::Four),
            8 => Some(Self::Eight),
            16 => Some(Self::Sixteen),
            _ => None,
        }
    }

    pub(crate) const fn bytes(self) -> u64 {
        match self {
            Self::Four => 4,
            Self::Eight => 8,
            Self::Sixteen => 16,
        }
    }
}

impl FunctionBuilder {
    /// Starts copying `size` bytes from global `src` to shared `dst`, the
    /// bytes past `copied` zero where it is given.
    pub(crate) fn copy_async(
        &mut self,
        size: CopySize,
        (dst, src): (Address, Address),
        copied: Option<Reg<B32>>,
    ) {
        self.push(Instr::CpAsync { size, dst, src, copied: copied.map(Reg::id) });
    }

    /// Closes the group of the copies started since the last.
    pub(crate) fn commit_copies(&mut self) {
        self.push(Instr::CpAsyncCommit);
    }

    /// Waits until at most `pending` committed groups are in flight.
    pub(crate) fn wait_copies(&mut self, pending: u32) {
        self.push(Instr::CpAsyncWait { pending });
    }
}
