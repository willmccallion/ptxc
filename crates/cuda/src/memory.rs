//! `DeviceBuffer`: device memory in bytes. No host memory is ever mapped
//! into the device's address space (see the crate docs).

use crate::context::Context;
use crate::error::CudaError;
use crate::sys::CuDevicePtr;
use ptxc_core::ByteSize;

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
