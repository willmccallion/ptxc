//! Streams.

use crate::context::Context;
use crate::error::CudaError;
use crate::sys::{CU_STREAM_DEFAULT, CuStream};

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
}
