//! Loading the driver once per process, and turning `CUresult`s into errors.

use crate::device::{Device, DeviceOrdinal};
use crate::error::CudaError;
use crate::sys::{Api, CUDA_SUCCESS, CuDevice, CuResult, Library};
use std::ffi::{CStr, c_char};
use std::fmt;
use std::sync::OnceLock;

/// The driver library; libcuda is loaded once and never unloaded (D24).
const LIBCUDA: &CStr = c"libcuda.so.1";

static API: OnceLock<Result<Api, CudaError>> = OnceLock::new();

/// The loaded CUDA driver.
#[derive(Clone, Copy)]
pub struct Driver {
    api: &'static Api,
}

impl fmt::Debug for Driver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Driver").finish_non_exhaustive()
    }
}

impl Driver {
    /// Loads `libcuda.so.1` and initialises it, once per process. Without a
    /// driver this is an error, never a panic.
    pub fn load() -> Result<Self, CudaError> {
        match API.get_or_init(|| open(LIBCUDA)) {
            Ok(api) => Ok(Self { api }),
            Err(error) => Err(error.clone()),
        }
    }

    /// The device numbered `ordinal`, its primary context retained.
    pub fn device(self, ordinal: DeviceOrdinal) -> Result<Device, CudaError> {
        let mut raw: CuDevice = 0;
        // SAFETY: `raw` is a valid place for the driver to write the device.
        let status = unsafe { (self.api.device_get)(&raw mut raw, i32::from(ordinal.get())) };
        check(self.api, "cuDeviceGet", status)?;
        Device::new(self.api, raw)
    }
}

/// Loads the driver from `library` and initialises it.
fn open(library: &CStr) -> Result<Api, CudaError> {
    let library = Library::open(library).map_err(CudaError::Load)?;
    let api = Api::resolve(&library).map_err(CudaError::Load)?;
    // SAFETY: cuInit with flags 0 has no preconditions.
    let status = unsafe { (api.init)(0) };
    check(&api, "cuInit", status)?;
    Ok(api)
}

/// `Ok` for `CUDA_SUCCESS`; otherwise the error `call` returned.
pub(crate) fn check(api: &Api, call: &'static str, status: CuResult) -> Result<(), CudaError> {
    if status == CUDA_SUCCESS { Ok(()) } else { Err(driver_error(api, call, status, None)) }
}

/// The error for `status`, named by the driver, with any `detail` it gave.
pub(crate) fn driver_error(
    api: &Api,
    call: &'static str,
    status: CuResult,
    detail: Option<String>,
) -> CudaError {
    CudaError::Driver { call, code: status, name: error_name(api, status), detail }
}

fn error_name(api: &Api, status: CuResult) -> String {
    let mut name: *const c_char = std::ptr::null();
    // SAFETY: `name` is a valid place for the driver to write a pointer.
    let lookup = unsafe { (api.get_error_name)(status, &raw mut name) };
    if lookup != CUDA_SUCCESS || name.is_null() {
        return format!("CUresult {status}");
    }
    // SAFETY: on success the driver points `name` at a static NUL-terminated
    // string.
    unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_driver_is_a_load_error_naming_the_library() {
        let Err(CudaError::Load(message)) = open(c"libptxc-no-such-driver.so.1") else {
            panic!("a missing library loaded")
        };

        assert!(message.contains("libptxc-no-such-driver.so.1"), "{message}");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn errors_are_named_by_the_driver_or_by_their_number() {
        let api = Driver::load().unwrap().api;

        assert_eq!(error_name(api, 2), "CUDA_ERROR_OUT_OF_MEMORY");
        assert_eq!(error_name(api, 123_456), "CUresult 123456");
    }
}
