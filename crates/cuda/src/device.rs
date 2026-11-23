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

/// A GPU.
#[derive(Clone, Copy)]
pub struct Device {
    api: &'static Api,
    raw: CuDevice,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device").field("raw", &self.raw).finish_non_exhaustive()
    }
}

impl Device {
    pub(crate) const fn new(api: &'static Api, raw: CuDevice) -> Self {
        Self { api, raw }
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

    /// The device's primary context, retained until every handle made from
    /// it is dropped.
    pub fn primary_context(&self) -> Result<Context, CudaError> {
        Context::retain_primary(self.api, self.raw)
    }
}
