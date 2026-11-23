//! `Target`: the description of one GPU architecture that scheduling and
//! code generation read.

use crate::limits::Limits;
use crate::mma::{MmaInstruction, MmaShape};
use crate::sm::{PtxIsaVersion, SmVersion};
use ptxc_core::DType;

/// A GPU architecture as data. Nothing outside `ptxc-target` names an
/// architecture; it reads one of these.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Target {
    pub(crate) sm: SmVersion,
    pub(crate) ptx_isa: PtxIsaVersion,
    pub(crate) limits: Limits,
    pub(crate) mma: &'static [MmaInstruction],
}

impl Target {
    /// The compute capability emitted modules target.
    pub const fn sm(&self) -> SmVersion {
        self.sm
    }

    /// The PTX ISA version emitted modules declare.
    pub const fn ptx_isa(&self) -> PtxIsaVersion {
        self.ptx_isa
    }

    /// The hardware limits.
    pub const fn limits(&self) -> &Limits {
        &self.limits
    }

    /// The `mma.sync` variants the target supports.
    pub const fn mma(&self) -> &'static [MmaInstruction] {
        self.mma
    }

    /// The variant of `shape` on `multiplicands` accumulating in
    /// `accumulator`, if the target has one.
    pub fn mma_instruction(
        &self,
        shape: MmaShape,
        multiplicands: DType,
        accumulator: DType,
    ) -> Option<MmaInstruction> {
        let wanted = MmaInstruction { shape, multiplicands, accumulator };
        self.mma.iter().copied().find(|&instruction| instruction == wanted)
    }
}
