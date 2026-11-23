//! A launch builder that requires grid and block dimensions before
//! launching, and checks the arguments against the kernel's parameters.

use crate::error::CudaError;
use crate::memory::{DeviceBuffer, DeviceSlice};
use crate::module::Function;
use crate::stream::Stream;
use ptxc_core::{ByteSize, KernelName};
use std::ffi::c_void;

/// A grid's extent in blocks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GridDim {
    x: u32,
    y: u32,
    z: u32,
}

impl GridDim {
    /// `x * y * z` blocks.
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

/// A block's extent in threads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockDim {
    x: u32,
    y: u32,
    z: u32,
}

impl BlockDim {
    /// `x * y * z` threads.
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

/// A launch dimension not set yet.
#[derive(Clone, Copy, Debug)]
pub struct Unset;

/// A kernel argument: a device buffer's address or a scalar.
#[derive(Clone, Copy, Debug)]
pub enum KernelArg<'a> {
    /// A buffer, passed as its device address.
    Buffer(&'a DeviceBuffer),
    /// Part of a buffer, passed as its first byte's device address.
    Slice(DeviceSlice<'a>),
    /// A `.u32` or `.s32` parameter's bits.
    U32(u32),
    /// A `.u64` or `.s64` parameter's bits.
    U64(u64),
    /// A `.f32` parameter.
    F32(f32),
}

impl<'a> From<&'a DeviceBuffer> for KernelArg<'a> {
    fn from(buffer: &'a DeviceBuffer) -> Self {
        Self::Buffer(buffer)
    }
}

impl<'a> From<DeviceSlice<'a>> for KernelArg<'a> {
    fn from(slice: DeviceSlice<'a>) -> Self {
        Self::Slice(slice)
    }
}

impl From<u32> for KernelArg<'_> {
    fn from(value: u32) -> Self {
        Self::U32(value)
    }
}

impl From<u64> for KernelArg<'_> {
    fn from(value: u64) -> Self {
        Self::U64(value)
    }
}

impl From<f32> for KernelArg<'_> {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

/// One argument's bytes, aligned for any parameter type.
#[repr(C, align(8))]
struct ArgSlot([u8; 8]);

impl KernelArg<'_> {
    const fn size(&self) -> ByteSize {
        match self {
            Self::Buffer(_) | Self::Slice(_) | Self::U64(_) => ByteSize::new(8),
            Self::U32(_) | Self::F32(_) => ByteSize::new(4),
        }
    }

    fn slot(&self) -> ArgSlot {
        let mut bytes = [0_u8; 8];
        match self {
            Self::Buffer(buffer) => bytes = buffer.address().to_ne_bytes(),
            Self::Slice(slice) => bytes = slice.address().to_ne_bytes(),
            Self::U64(value) => bytes = value.to_ne_bytes(),
            Self::U32(value) => bytes[..4].copy_from_slice(&value.to_ne_bytes()),
            Self::F32(value) => bytes[..4].copy_from_slice(&value.to_ne_bytes()),
        }
        ArgSlot(bytes)
    }
}

/// A kernel launch being built: `G` and `B` become `GridDim` and `BlockDim`
/// once set, and only then can it launch.
#[derive(Debug)]
#[must_use = "a launch does nothing until `.launch(&stream)`"]
pub struct Launch<'a, G, B> {
    function: &'a Function,
    grid: G,
    block: B,
    dynamic_smem: ByteSize,
    args: Vec<KernelArg<'a>>,
}

impl Function {
    /// Starts a launch of this kernel.
    pub const fn launch(&self) -> Launch<'_, Unset, Unset> {
        Launch {
            function: self,
            grid: Unset,
            block: Unset,
            dynamic_smem: ByteSize::new(0),
            args: Vec::new(),
        }
    }
}

impl<'a, B> Launch<'a, Unset, B> {
    /// Sets the grid.
    pub fn grid(self, grid: GridDim) -> Launch<'a, GridDim, B> {
        let Self { function, block, dynamic_smem, args, .. } = self;
        Launch { function, grid, block, dynamic_smem, args }
    }
}

impl<'a, G> Launch<'a, G, Unset> {
    /// Sets the block.
    pub fn block(self, block: BlockDim) -> Launch<'a, G, BlockDim> {
        let Self { function, grid, dynamic_smem, args, .. } = self;
        Launch { function, grid, block, dynamic_smem, args }
    }
}

impl<'a, G, B> Launch<'a, G, B> {
    /// Sets the dynamic shared memory per block (0 by default).
    pub const fn dynamic_smem(mut self, bytes: ByteSize) -> Self {
        self.dynamic_smem = bytes;
        self
    }

    /// Appends the next argument.
    pub fn arg(mut self, arg: impl Into<KernelArg<'a>>) -> Self {
        self.args.push(arg.into());
        self
    }
}

impl Launch<'_, GridDim, BlockDim> {
    /// Queues the kernel on `stream`. The arguments must match the kernel's
    /// parameters in number and size, and every buffer and the stream must
    /// share the kernel's context.
    pub fn launch(self, stream: &Stream) -> Result<(), CudaError> {
        let context = self.function.context();
        if !context.is(stream.context()) {
            return Err(CudaError::ForeignContext("the stream"));
        }
        let foreign = |arg: &KernelArg<'_>| match arg {
            KernelArg::Buffer(buffer) => !context.is(buffer.context()),
            KernelArg::Slice(slice) => !context.is(slice.context()),
            KernelArg::U32(_) | KernelArg::U64(_) | KernelArg::F32(_) => false,
        };
        if self.args.iter().any(foreign) {
            return Err(CudaError::ForeignContext("a buffer argument"));
        }
        let sizes: Vec<ByteSize> = self.args.iter().map(KernelArg::size).collect();
        check_args(self.function.name(), self.function.parameter_sizes(), &sizes)?;
        let smem = u32::try_from(self.dynamic_smem.get())
            .map_err(|_| CudaError::TooLarge(self.dynamic_smem))?;
        let mut slots: Vec<ArgSlot> = self.args.iter().map(KernelArg::slot).collect();
        let mut pointers: Vec<*mut c_void> =
            slots.iter_mut().map(|slot| slot.0.as_mut_ptr().cast()).collect();
        let (grid, block) = (self.grid, self.block);
        context.call("cuLaunchKernel", |api| {
            // SAFETY: the function, stream and buffers are alive and share
            // the context; `pointers` holds one pointer per parameter, each
            // to at least the parameter's size of bytes in `slots`, which
            // outlive the call (the driver copies them at launch). What the
            // kernel does with device memory cannot reach host memory, as
            // none is mapped into the device's address space.
            unsafe {
                (api.launch_kernel)(
                    self.function.raw(),
                    grid.x,
                    grid.y,
                    grid.z,
                    block.x,
                    block.y,
                    block.z,
                    smem,
                    stream.raw(),
                    pointers.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
        })
    }
}

/// Checks that `args` match `params` in number and in each size.
fn check_args(
    kernel: &KernelName,
    params: &[ByteSize],
    args: &[ByteSize],
) -> Result<(), CudaError> {
    if params.len() != args.len() {
        return Err(CudaError::ArgumentCount {
            kernel: kernel.clone(),
            expected: params.len(),
            actual: args.len(),
        });
    }
    for (index, (&expected, &actual)) in params.iter().zip(args).enumerate() {
        if expected != actual {
            return Err(CudaError::ArgumentSize {
                kernel: kernel.clone(),
                index,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
