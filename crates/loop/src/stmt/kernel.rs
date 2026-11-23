//! `LoweredKernel`: parameters, launch shape, shared-memory size and body.

use super::Stmt;
use super::buffer::{Access, Buffer, BufferId};
use crate::compute::VarName;
use crate::expr::{ExprArena, VarId};
use crate::schedule::ThreadAxis;
use ptxc_core::{IndexVec, KernelName};
use ptxc_target::{BlockCount, SmemBytes, ThreadsPerBlock};

/// What a variable of a lowered kernel stands for.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum KernelVar {
    /// A loop variable, bound by a `For`.
    Loop(VarName),
    /// A grid or block index, bound everywhere in the kernel.
    Thread {
        /// The loop it replaces.
        name: VarName,
        /// The axis.
        axis: ThreadAxis,
    },
    /// An integer read from memory, bound by a `LoadIndex`.
    Loaded(VarName),
}

impl KernelVar {
    /// The name it prints with.
    pub const fn name(&self) -> &VarName {
        match self {
            Self::Loop(name) | Self::Thread { name, .. } | Self::Loaded(name) => name,
        }
    }
}

/// A global buffer passed to the kernel, in parameter order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct KernelParam {
    /// The buffer.
    pub buffer: BufferId,
    /// Whether the kernel reads or writes it.
    pub access: Access,
}

/// The grid and block a kernel launches with.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LaunchShape {
    /// Blocks along x, y and z.
    pub grid: [BlockCount; 3],
    /// Threads along x, y and z.
    pub block: [ThreadsPerBlock; 3],
}

/// A kernel as an explicit loop nest, ready for code generation.
#[derive(Clone, Debug)]
pub struct LoweredKernel {
    pub(crate) name: KernelName,
    pub(crate) params: Vec<KernelParam>,
    pub(crate) buffers: IndexVec<BufferId, Buffer>,
    pub(crate) vars: IndexVec<VarId, KernelVar>,
    pub(crate) launch: LaunchShape,
    pub(crate) smem: SmemBytes,
    pub(crate) body: Stmt,
    pub(crate) exprs: ExprArena,
}

impl LoweredKernel {
    /// The kernel's `.entry` name.
    pub const fn name(&self) -> &KernelName {
        &self.name
    }

    /// The global buffers it takes: the inputs, then the outputs.
    pub fn params(&self) -> &[KernelParam] {
        &self.params
    }

    /// Every buffer it uses.
    pub const fn buffers(&self) -> &IndexVec<BufferId, Buffer> {
        &self.buffers
    }

    /// What each variable stands for.
    pub const fn vars(&self) -> &IndexVec<VarId, KernelVar> {
        &self.vars
    }

    /// The grid and block.
    pub const fn launch(&self) -> LaunchShape {
        self.launch
    }

    /// The shared memory its blocks use.
    pub const fn smem(&self) -> SmemBytes {
        self.smem
    }

    /// The body every thread runs.
    pub const fn body(&self) -> &Stmt {
        &self.body
    }

    /// The index expressions it reads.
    pub const fn exprs(&self) -> &ExprArena {
        &self.exprs
    }
}
