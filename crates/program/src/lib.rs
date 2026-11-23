//! The compiled model: kernels, launch configuration, buffer plan and
//! execution steps, independent of any IR, and its file format.

mod buffer_plan;
mod format;
mod kernel;
mod program;
mod step;

pub use buffer_plan::{BUFFER_ALIGNMENT, Buffer, BufferId, InputIndex, Location, OutputIndex};
pub use format::{FORMAT_VERSION, FormatError, load, save};
pub use kernel::{KernelId, KernelSpec, LaunchConfig};
pub use program::{IoBinding, Program, ProgramError, ProgramParts};
pub use step::Step;
