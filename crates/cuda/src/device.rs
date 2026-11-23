//! Devices.

use crate::context::Context;
use crate::driver::check;
use crate::error::CudaError;
use crate::sys::{
    Api, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CuDevice,
};
use std::ffi::c_int;
use std::fmt;

/// A device's number among the driver's devices, from 0.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DeviceOrdinal(u16);

impl DeviceOrdinal {
    /// Wraps an ordinal.
    pub const fn new(ordinal: u16) -> Self {
        Self(ordinal)
    }

    /// The ordinal.
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// A compute capability, such as 8.9 for Ada.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ComputeCapability {
    /// The major version.
    pub major: u8,
    /// The minor version.
    pub minor: u8,
}

/// A GPU, holding its primary context retained for as long as it lives.
// NOTE: the driver destroys a primary context when its last retain is
// released, and creating one costs far more than a launch. As the CUDA
// runtime API keeps it alive until a reset, a device keeps it, so programs
// loaded and dropped one after another share one context.
#[derive(Clone)]
pub struct Device {
    api: &'static Api,
    raw: CuDevice,
    context: Context,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device").field("raw", &self.raw).finish_non_exhaustive()
    }
}

impl Device {
    pub(crate) fn new(api: &'static Api, raw: CuDevice) -> Result<Self, CudaError> {
        Ok(Self { api, raw, context: Context::retain_primary(api, raw)? })
    }

    /// The device's compute capability, such as 8.9.
    pub fn compute_capability(&self) -> Result<ComputeCapability, CudaError> {
        let major = self.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = self.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok(ComputeCapability { major, minor })
    }

    /// The integer device attribute `attribute`, which the driver defines as
    /// a small non-negative number.
    fn attribute(&self, attribute: c_int) -> Result<u8, CudaError> {
        let mut value: c_int = 0;
        // SAFETY: `value` is a valid place for the driver to write, and
        // `raw` is a device the driver returned.
        let status =
            unsafe { (self.api.device_get_attribute)(&raw mut value, attribute, self.raw) };
        check(self.api, "cuDeviceGetAttribute", status)?;
        u8::try_from(value).map_err(|_| CudaError::Attribute { attribute, value })
    }

    /// The device's primary context.
    pub fn primary_context(&self) -> Context {
        self.context.clone()
    }
}

#[cfg(all(test, feature = "gpu"))]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::DeviceOrdinal;
    use crate::driver::Driver;
    use crate::sys::{Api, CuDevice};
    use std::ffi::{c_int, c_uint};

    fn is_context_active(api: &Api, raw: CuDevice) -> bool {
        let mut flags: c_uint = 0;
        let mut active: c_int = 0;
        // SAFETY: `flags` and `active` are valid places for the driver to
        // write, and `raw` is a device the driver returned.
        let status = unsafe { (api.primary_ctx_get_state)(raw, &raw mut flags, &raw mut active) };
        assert_eq!(status, crate::sys::CUDA_SUCCESS);
        active != 0
    }

    #[test]
    fn the_primary_context_lives_exactly_as_long_as_the_device() {
        let device = Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap();
        let (api, raw) = (device.api, device.raw);

        drop(device.primary_context());
        let active_with_device = is_context_active(api, raw);
        drop(device);
        let active_after_device = is_context_active(api, raw);

        assert!(active_with_device);
        assert!(!active_after_device);
    }
}
