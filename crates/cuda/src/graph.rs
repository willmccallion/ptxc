//! Stream capture: a `GraphCapture` records the work queued on a stream
//! into a `CapturedGraph`, which instantiates into a `GraphExec` that
//! replays it all with one launch.

use crate::context::Context;
use crate::error::CudaError;
use crate::stream::Stream;
use crate::sys::{CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, CuGraph, CuGraphExec};
use std::mem::ManuallyDrop;

/// A stream recording the work queued on it, rather than running it,
/// until `end`. Dropping it ends the capture and discards what it
/// recorded, so the stream never stays capturing.
#[derive(Debug)]
#[must_use = "a capture records nothing usable unless ended"]
pub struct GraphCapture<'s> {
    stream: &'s Stream,
}

impl Stream {
    /// Starts recording the work queued on this stream into a graph, as
    /// `cuStreamBeginCapture` does.
    ///
    /// The capture is thread-local: only this thread is barred, until the
    /// capture ends, from calls a capture cannot hold, such as synchronous
    /// copies; other threads sharing the context carry on.
    pub fn begin_capture(&self) -> Result<GraphCapture<'_>, CudaError> {
        self.context().call("cuStreamBeginCapture", |api| {
            // SAFETY: the stream is live in this context.
            unsafe { (api.stream_begin_capture)(self.raw(), CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) }
        })?;
        Ok(GraphCapture { stream: self })
    }
}

impl<'s> GraphCapture<'s> {
    /// The stream being recorded: work queued on it is recorded, not run.
    pub const fn stream(&self) -> &'s Stream {
        self.stream
    }

    /// Ends the capture: the graph of the work queued while it ran.
    pub fn end(self) -> Result<CapturedGraph, CudaError> {
        let capture = ManuallyDrop::new(self);
        let raw = capture.end_capture()?;
        Ok(CapturedGraph { context: capture.stream.context().clone(), raw })
    }

    /// Ends the capture and returns the graph it recorded.
    fn end_capture(&self) -> Result<CuGraph, CudaError> {
        let mut raw = CuGraph::null();
        self.stream.context().call("cuStreamEndCapture", |api| {
            // SAFETY: the stream is capturing, from `begin_capture`, and
            // `raw` is a valid place for the driver to write the graph.
            unsafe { (api.stream_end_capture)(self.stream.raw(), &raw mut raw) }
        })?;
        Ok(raw)
    }
}

impl Drop for GraphCapture<'_> {
    fn drop(&mut self) {
        // A failed end cannot be reported from `drop`; the driver ends the
        // capture whether or not it can return a graph.
        if let Ok(raw) = self.end_capture() {
            drop(CapturedGraph { context: self.stream.context().clone(), raw });
        }
    }
}

/// The work a capture recorded, destroyed on drop.
#[derive(Debug)]
pub struct CapturedGraph {
    context: Context,
    raw: CuGraph,
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        // A failed destroy cannot be reported from `drop`.
        let _failed = self.context.call("cuGraphDestroy", |api| {
            // SAFETY: the graph came from a capture in this context and is
            // destroyed exactly once, here.
            unsafe { (api.graph_destroy)(self.raw) }
        });
    }
}

impl CapturedGraph {
    /// The graph made executable, as `cuGraphInstantiate` does.
    pub fn instantiate(&self) -> Result<GraphExec, CudaError> {
        let mut raw = CuGraphExec::null();
        self.context.call("cuGraphInstantiate", |api| {
            // SAFETY: the graph is live in this context, and `raw` is a
            // valid place for the driver to write the executable graph.
            unsafe { (api.graph_instantiate)(&raw mut raw, self.raw, 0) }
        })?;
        Ok(GraphExec { context: self.context.clone(), raw })
    }
}

/// A graph ready to replay, destroyed on drop.
///
/// A replay launches the recorded kernels and copies on the buffers they
/// were given when recorded. A buffer freed since leaves them writing
/// device memory it no longer owns, which only the device sees, as with
/// any wrong kernel: this crate maps no host memory into the device.
#[derive(Debug)]
pub struct GraphExec {
    context: Context,
    raw: CuGraphExec,
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        // A failed destroy cannot be reported from `drop`.
        let _failed = self.context.call("cuGraphExecDestroy", |api| {
            // SAFETY: the executable graph was instantiated in this context
            // and is destroyed exactly once, here.
            unsafe { (api.graph_exec_destroy)(self.raw) }
        });
    }
}

impl GraphExec {
    /// Queues the whole graph on `stream`, after the work queued before it.
    pub fn launch(&self, stream: &Stream) -> Result<(), CudaError> {
        if !self.context.is(stream.context()) {
            return Err(CudaError::ForeignContext("a graph's stream"));
        }
        self.context.call("cuGraphLaunch", |api| {
            // SAFETY: the executable graph and the stream are live in this
            // context.
            unsafe { (api.graph_launch)(self.raw, stream.raw()) }
        })
    }
}
