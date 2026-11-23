//! Events and elapsed-time measurement.

use crate::context::Context;
use crate::error::CudaError;
use crate::stream::Stream;
use crate::sys::{CU_EVENT_DEFAULT, CuEvent};
use ptxc_core::Microseconds;

/// A marker in a stream that records when the work before it finished.
#[derive(Debug)]
pub struct Event {
    context: Context,
    raw: CuEvent,
}

impl Drop for Event {
    fn drop(&mut self) {
        // A failed destroy cannot be reported from `drop`.
        let _failed = self.context.call("cuEventDestroy", |api| {
            // SAFETY: the event was created in this context and is destroyed
            // exactly once, here.
            unsafe { (api.event_destroy)(self.raw) }
        });
    }
}

impl Event {
    pub(crate) fn create(context: &Context) -> Result<Self, CudaError> {
        let mut raw = CuEvent::null();
        context.call("cuEventCreate", |api| {
            // SAFETY: `raw` is a valid place for the driver to write.
            unsafe { (api.event_create)(&raw mut raw, CU_EVENT_DEFAULT) }
        })?;
        Ok(Self { context: context.clone(), raw })
    }

    /// Records the event after the work queued on `stream` so far.
    pub fn record(&self, stream: &Stream) -> Result<(), CudaError> {
        if !self.context.is(stream.context()) {
            return Err(CudaError::ForeignContext("the stream"));
        }
        self.context.call("cuEventRecord", |api| {
            // SAFETY: the event and stream are alive and share a context.
            unsafe { (api.event_record)(self.raw, stream.raw()) }
        })
    }

    /// Waits until the work before the event's last record has finished.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.context.call("cuEventSynchronize", |api| {
            // SAFETY: the event is alive for as long as `self`.
            unsafe { (api.event_synchronize)(self.raw) }
        })
    }

    /// The time between `start` and this event, both recorded and finished.
    pub fn elapsed_since(&self, start: &Self) -> Result<Microseconds, CudaError> {
        if !self.context.is(&start.context) {
            return Err(CudaError::ForeignContext("the start event"));
        }
        let mut milliseconds = 0.0_f32;
        self.context.call("cuEventElapsedTime", |api| {
            // SAFETY: both events are alive and share a context, and
            // `milliseconds` is a valid place for the driver to write.
            unsafe { (api.event_elapsed_time)(&raw mut milliseconds, start.raw, self.raw) }
        })?;
        Ok(Microseconds::from_millis(milliseconds))
    }
}
