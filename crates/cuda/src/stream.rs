//! Streams.

use crate::context::Context;
use crate::error::CudaError;
use crate::memory::DeviceSlice;
use crate::sys::{CU_STREAM_DEFAULT, CuStream};
use std::ffi::c_void;

/// A queue of device work, destroyed on drop.
#[derive(Debug)]
pub struct Stream {
    context: Context,
    raw: CuStream,
}

impl Drop for Stream {
    fn drop(&mut self) {
        // A failed destroy cannot be reported from `drop`.
        let _failed = self.context.call("cuStreamDestroy", |api| {
            // SAFETY: the stream was created in this context and is
            // destroyed exactly once, here.
            unsafe { (api.stream_destroy)(self.raw) }
        });
    }
}

impl Stream {
    pub(crate) fn create(context: &Context) -> Result<Self, CudaError> {
        let mut raw = CuStream::null();
        context.call("cuStreamCreate", |api| {
            // SAFETY: `raw` is a valid place for the driver to write.
            unsafe { (api.stream_create)(&raw mut raw, CU_STREAM_DEFAULT) }
        })?;
        Ok(Self { context: context.clone(), raw })
    }

    /// Queues a copy of `from`'s bytes into `to`, which must be as long,
    /// after the work queued before it.
    pub fn copy(&self, from: DeviceSlice<'_>, to: DeviceSlice<'_>) -> Result<(), CudaError> {
        if !self.context.is(from.context()) || !self.context.is(to.context()) {
            return Err(CudaError::ForeignContext("a copied buffer"));
        }
        if from.size() != to.size() {
            return Err(CudaError::SliceSizes { from: from.size(), to: to.size() });
        }
        let bytes =
            usize::try_from(from.size().get()).map_err(|_| CudaError::TooLarge(from.size()))?;
        if bytes == 0 {
            return Ok(());
        }
        self.context.call("cuMemcpyDtoDAsync", |api| {
            // SAFETY: both slices lie within live allocations of this
            // context, each `bytes` long, and the copy touches only device
            // memory, none of which is host memory mapped into the device.
            unsafe { (api.memcpy_dtod_async)(to.address(), from.address(), bytes, self.raw) }
        })
    }

    /// Waits for everything queued on the stream to finish.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.context.call("cuStreamSynchronize", |api| {
            // SAFETY: the stream is alive for as long as `self`.
            unsafe { (api.stream_synchronize)(self.raw) }
        })
    }

    pub(crate) const fn context(&self) -> &Context {
        &self.context
    }

    pub(crate) const fn raw(&self) -> CuStream {
        self.raw
    }

    /// The driver's `CUstream` handle, for passing to another CUDA library;
    /// valid while the stream lives.
    pub const fn handle(&self) -> *mut c_void {
        self.raw.0
    }
}
