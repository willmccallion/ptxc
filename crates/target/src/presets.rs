//! Built-in targets for known architectures.

use crate::limits::{
    BlockCount, BlockLimits, GridLimits, Limits, RegisterCount, SmLimits, SmemBytes, ThreadCount,
    ThreadsPerBlock,
};
use crate::mma;
use crate::sm::{PtxIsaVersion, SmVersion};
use crate::target::Target;
use thiserror::Error;

/// A compute capability with no preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("ptxc has no target for {0}; supported: {sm_89}", sm_89 = SmVersion::SM_89)]
pub struct UnsupportedSm(pub SmVersion);

/// `sm_89`, with the PTX ISA version that introduced it, so its modules load
/// on any driver that knows `sm_89` (D36). The limits are compute capability
/// 8.9's, as an RTX 4060 Ti's driver reports them; registers per thread is
/// from the CUDA programming guide, as no device attribute gives it.
const SM_89: Target = Target {
    sm: SmVersion::SM_89,
    ptx_isa: PtxIsaVersion::new(7, 8),
    limits: Limits {
        warp_size: ThreadCount::new(32),
        registers_per_thread: RegisterCount::new(255),
        block: BlockLimits {
            threads: ThreadsPerBlock::new(1024),
            extents: [
                ThreadsPerBlock::new(1024),
                ThreadsPerBlock::new(1024),
                ThreadsPerBlock::new(64),
            ],
            registers: RegisterCount::new(65_536),
            smem: SmemBytes::new(49_152),
            smem_opt_in: SmemBytes::new(101_376),
            smem_reserved: SmemBytes::new(1024),
        },
        sm: SmLimits {
            threads: ThreadCount::new(1536),
            blocks: BlockCount::new(24),
            registers: RegisterCount::new(65_536),
            smem: SmemBytes::new(102_400),
        },
        grid: GridLimits {
            extents: [
                BlockCount::new(2_147_483_647),
                BlockCount::new(65_535),
                BlockCount::new(65_535),
            ],
        },
    },
    mma: mma::SM_89,
};

/// The built-in target for compute capability `sm`.
pub const fn preset(sm: SmVersion) -> Result<Target, UnsupportedSm> {
    match sm {
        SmVersion::SM_89 => Ok(SM_89),
        other => Err(UnsupportedSm(other)),
    }
}
