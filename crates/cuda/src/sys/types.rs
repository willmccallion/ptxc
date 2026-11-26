//! Driver API handle and enum types, as `cuda.h` declares them.

use std::ffi::{c_int, c_uint, c_void};

/// `CUresult`: 0 is success, anything else names an error.
pub(crate) type CuResult = c_int;

/// `CUDA_SUCCESS`.
pub(crate) const CUDA_SUCCESS: CuResult = 0;

/// `CUDA_ERROR_INVALID_VALUE`, which `cuFuncGetParamInfo` returns past the
/// last parameter.
pub(crate) const CUDA_ERROR_INVALID_VALUE: CuResult = 1;

/// `CUdevice`: a device ordinal as the driver numbers it.
pub(crate) type CuDevice = c_int;

/// `CUdeviceptr`: a device address.
pub(crate) type CuDevicePtr = u64;

/// `CUjit_option`.
pub(crate) type CuJitOption = c_int;

/// `CU_JIT_ERROR_LOG_BUFFER`: where the JIT writes its error log.
pub(crate) const CU_JIT_ERROR_LOG_BUFFER: CuJitOption = 5;

/// `CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`: that buffer's size.
pub(crate) const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: CuJitOption = 6;

/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`.
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;

/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`.
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

/// `CU_STREAM_DEFAULT`: a stream that synchronizes with the legacy default
/// stream, so the synchronous copies (which run there) are ordered with it.
pub(crate) const CU_STREAM_DEFAULT: c_uint = 0;

/// `CU_EVENT_DEFAULT`: an event that records timing.
pub(crate) const CU_EVENT_DEFAULT: c_uint = 0;

/// `CUstreamCaptureMode`.
pub(crate) type CuStreamCaptureMode = c_int;

/// `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`: while this thread captures, only
/// it is barred from calls a capture cannot hold; other threads' captures
/// and calls are left alone.
pub(crate) const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: CuStreamCaptureMode = 1;

macro_rules! handles {
    ($($(#[$doc:meta])* $name:ident),* $(,)?) => {$(
        $(#[$doc])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) struct $name(pub(crate) *mut c_void);

        impl $name {
            /// The null handle, for out-parameters the driver fills in.
            pub(crate) const fn null() -> Self {
                Self(std::ptr::null_mut())
            }
        }

        // SAFETY: the driver API is thread-safe, and its handles may be used
        // from any thread; this type only carries the handle value.
        unsafe impl Send for $name {}
        // SAFETY: as above; a shared handle is only passed back to the driver.
        unsafe impl Sync for $name {}
    )*};
}

handles! {
    /// `CUcontext`.
    CuContext,
    /// `CUmodule`.
    CuModule,
    /// `CUfunction`.
    CuFunction,
    /// `CUstream`.
    CuStream,
    /// `CUevent`.
    CuEvent,
    /// `CUgraph`.
    CuGraph,
    /// `CUgraphExec`.
    CuGraphExec,
}
