//! `DeviceBuffer`: device memory in bytes. No host memory is ever mapped
//! into the device's address space (see the crate docs).

use crate::context::Context;
use crate::error::CudaError;
use crate::sys::CuDevicePtr;
use ptxc_core::{ByteOffset, ByteSize};

/// A device address, for passing to another CUDA library.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DeviceAddress(u64);

impl DeviceAddress {
    /// The address.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A device allocation, freed on drop.
#[derive(Debug)]
pub struct DeviceBuffer {
    context: Context,
    address: CuDevicePtr,
    size: ByteSize,
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // A failed free cannot be reported from `drop`; the memory stays
        // allocated until its context is released.
        let _failed = self.context.call("cuMemFree", |api| {
            // SAFETY: `address` was allocated in this context and is freed
            // exactly once, here.
            unsafe { (api.mem_free)(self.address) }
        });
    }
}

impl DeviceBuffer {
    pub(crate) fn alloc(context: &Context, size: ByteSize) -> Result<Self, CudaError> {
        let bytes = to_usize(size)?;
        let mut address: CuDevicePtr = 0;
        context.call("cuMemAlloc", |api| {
            // SAFETY: `address` is a valid place for the driver to write.
            unsafe { (api.mem_alloc)(&raw mut address, bytes) }
        })?;
        Ok(Self { context: context.clone(), address, size })
    }

    /// The buffer's size.
    pub const fn size(&self) -> ByteSize {
        self.size
    }

    /// The buffer's first byte's device address, valid while the buffer
    /// lives.
    pub const fn device_address(&self) -> DeviceAddress {
        DeviceAddress(self.address)
    }

    /// The whole buffer as a slice.
    pub const fn all(&self) -> DeviceSlice<'_> {
        DeviceSlice { buffer: self, offset: ByteOffset::new(0), size: self.size }
    }

    /// The `size` bytes from `offset`, if they lie within the buffer.
    pub const fn slice(
        &self,
        offset: ByteOffset,
        size: ByteSize,
    ) -> Result<DeviceSlice<'_>, CudaError> {
        match offset.get().checked_add(size.get()) {
            Some(end) if end <= self.size.get() => Ok(DeviceSlice { buffer: self, offset, size }),
            _ => Err(CudaError::SliceBounds { offset, size, buffer: self.size }),
        }
    }

    /// Overwrites the whole buffer with `bytes`, which must be its size. The
    /// copy runs on the legacy default stream, after work already queued on
    /// ptxc's streams, and returns when it is done.
    pub fn copy_from_host(&mut self, bytes: &[u8]) -> Result<(), CudaError> {
        self.check_host_size(bytes)?;
        self.context.call("cuMemcpyHtoD", |api| {
            // SAFETY: `bytes` and the buffer are both `bytes.len()` long, and
            // the copy completes before the call returns.
            unsafe { (api.memcpy_htod)(self.address, bytes.as_ptr().cast(), bytes.len()) }
        })
    }

    /// Reads the whole buffer into `bytes`, which must be its size, after
    /// work already queued on ptxc's streams.
    pub fn copy_to_host(&self, bytes: &mut [u8]) -> Result<(), CudaError> {
        self.check_host_size(bytes)?;
        self.context.call("cuMemcpyDtoH", |api| {
            // SAFETY: `bytes` and the buffer are both `bytes.len()` long,
            // `bytes` is borrowed mutably, and the copy completes before the
            // call returns.
            unsafe { (api.memcpy_dtoh)(bytes.as_mut_ptr().cast(), self.address, bytes.len()) }
        })
    }

    fn check_host_size(&self, bytes: &[u8]) -> Result<(), CudaError> {
        let host = ByteSize::new(bytes.len() as u64);
        if host == self.size {
            Ok(())
        } else {
            Err(CudaError::CopySize { host, device: self.size })
        }
    }

    pub(crate) const fn address(&self) -> CuDevicePtr {
        self.address
    }

    pub(crate) const fn context(&self) -> &Context {
        &self.context
    }
}

/// `size` as the driver's `size_t`.
pub(crate) fn to_usize(size: ByteSize) -> Result<usize, CudaError> {
    usize::try_from(size.get()).map_err(|_| CudaError::TooLarge(size))
}

/// A byte range within a device buffer: a kernel argument, or an end of a
/// device-to-device copy.
#[derive(Clone, Copy, Debug)]
pub struct DeviceSlice<'a> {
    buffer: &'a DeviceBuffer,
    offset: ByteOffset,
    size: ByteSize,
}

impl DeviceSlice<'_> {
    /// The slice's length.
    pub const fn size(&self) -> ByteSize {
        self.size
    }

    /// The device address of the slice's first byte, within its buffer by
    /// construction.
    pub(crate) const fn address(&self) -> CuDevicePtr {
        self.buffer.address() + self.offset.get()
    }

    pub(crate) const fn context(&self) -> &Context {
        self.buffer.context()
    }
}
