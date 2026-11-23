//! Loads a `Program`: its modules, the arena of intermediates and its
//! constants, once.

use crate::error::RuntimeError;
use ptxc_core::{ByteSize, IndexVec};
use ptxc_cuda::{Context, Device, DeviceBuffer, Function};
use ptxc_program::{KernelId, Program};

/// A program on a device, ready to run.
#[derive(Debug)]
pub struct LoadedProgram {
    pub(crate) program: Program,
    pub(crate) context: Context,
    pub(crate) functions: IndexVec<KernelId, Function>,
    pub(crate) arena: DeviceBuffer,
    pub(crate) constants: DeviceBuffer,
}

/// `program` loaded onto `device`, whose compute capability must be the
/// program's target.
pub fn load(program: Program, device: &Device) -> Result<LoadedProgram, RuntimeError> {
    let capability = device.compute_capability()?;
    let target = program.target();
    if (capability.major, capability.minor) != (target.major(), target.minor()) {
        return Err(RuntimeError::Target { program: target, device: capability });
    }
    let context = device.primary_context()?;
    let functions = program
        .kernels()
        .iter()
        .map(|kernel| context.load_ptx(&kernel.ptx)?.function(&kernel.name))
        .collect::<Result<_, _>>()?;
    let arena = allocation(&context, program.arena())?;
    let mut constants = allocation(&context, byte_size(program.constants().len())?)?;
    if !program.constants().is_empty() {
        constants.copy_from_host(program.constants())?;
    }
    Ok(LoadedProgram { program, context, functions, arena, constants })
}

/// A device allocation of `size` bytes.
// NOTE: the driver refuses zero-byte allocations, so an empty arena or blob
// takes one byte nothing reads.
pub(crate) fn allocation(context: &Context, size: ByteSize) -> Result<DeviceBuffer, RuntimeError> {
    Ok(context.alloc(ByteSize::new(size.get().max(1)))?)
}

/// `length` host bytes as a device size.
pub(crate) fn byte_size(length: usize) -> Result<ByteSize, RuntimeError> {
    u64::try_from(length).map(ByteSize::new).map_err(|_| RuntimeError::TooLarge)
}
