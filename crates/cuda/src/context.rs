//! The primary context, shared by every handle made from it.

use crate::driver::check;
use crate::error::CudaError;
use crate::event::Event;
use crate::memory::DeviceBuffer;
use crate::module::Module;
use crate::stream::Stream;
use crate::sys::{Api, CuContext, CuDevice, CuResult};
use ptxc_core::ByteSize;
use std::fmt;
use std::sync::Arc;

/// A device's primary context. Clones share it; it is released when the
/// last clone, and every handle made from it, is dropped.
#[derive(Clone, Debug)]
pub struct Context {
    inner: Arc<ContextInner>,
}

struct ContextInner {
    api: &'static Api,
    device: CuDevice,
    raw: CuContext,
}

impl fmt::Debug for ContextInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextInner").field("device", &self.device).finish_non_exhaustive()
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        // SAFETY: the primary context of `device` was retained when this
        // value was made, and is released exactly once, here.
        let status = unsafe { (self.api.primary_ctx_release)(self.device) };
        // A failed release cannot be reported from `drop`; the context
        // stays retained until the process exits.
        let _failed = check(self.api, "cuDevicePrimaryCtxRelease", status);
    }
}

impl Context {
    pub(crate) fn retain_primary(api: &'static Api, device: CuDevice) -> Result<Self, CudaError> {
        let mut raw = CuContext::null();
        // SAFETY: `raw` is a valid place for the driver to write the context.
        let status = unsafe { (api.primary_ctx_retain)(&raw mut raw, device) };
        check(api, "cuDevicePrimaryCtxRetain", status)?;
        Ok(Self { inner: Arc::new(ContextInner { api, device, raw }) })
    }

    pub(crate) fn api(&self) -> &'static Api {
        self.inner.api
    }

    /// Makes this context current on the calling thread, then runs `call`,
    /// a driver function named `name`.
    pub(crate) fn call(
        &self,
        name: &'static str,
        call: impl FnOnce(&Api) -> CuResult,
    ) -> Result<(), CudaError> {
        self.make_current()?;
        check(self.inner.api, name, call(self.inner.api))
    }

    /// Makes this context current on the calling thread.
    pub(crate) fn make_current(&self) -> Result<(), CudaError> {
        // SAFETY: the context is retained for as long as `self` lives.
        let status = unsafe { (self.inner.api.ctx_set_current)(self.inner.raw) };
        check(self.inner.api, "cuCtxSetCurrent", status)
    }

    /// Whether `other` is the same driver context, which every retain of a
    /// device's primary context is.
    pub(crate) fn is(&self, other: &Self) -> bool {
        self.inner.raw == other.inner.raw
    }

    /// JIT-compiles `ptx` into a module.
    pub fn load_ptx(&self, ptx: &str) -> Result<Module, CudaError> {
        Module::load(self, ptx)
    }

    /// Allocates `size` bytes of device memory.
    pub fn alloc(&self, size: ByteSize) -> Result<DeviceBuffer, CudaError> {
        DeviceBuffer::alloc(self, size)
    }

    /// Creates a stream.
    pub fn stream(&self) -> Result<Stream, CudaError> {
        Stream::create(self)
    }

    /// Creates an event that records timing.
    pub fn event(&self) -> Result<Event, CudaError> {
        Event::create(self)
    }
}
