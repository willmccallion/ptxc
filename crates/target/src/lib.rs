//! GPU target description as data: SM version, PTX ISA version, hardware limits and the supported mma instructions.

mod limits;
mod mma;
mod presets;
mod sm;
mod target;

pub use limits::{
    BlockCount, BlockLimits, GridLimits, Limits, RegisterCount, SmLimits, SmemBytes, ThreadCount,
    ThreadsPerBlock,
};
pub use mma::{MmaInstruction, MmaShape};
pub use presets::{UnsupportedSm, preset};
pub use sm::{PtxIsaVersion, SmVersion};
pub use target::Target;
