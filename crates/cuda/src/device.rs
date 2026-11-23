//! Devices.

use crate::context::Context;
use crate::error::CudaError;
use crate::sys::{Api, CuDevice};
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

    /// The device's primary context, retained until every handle made from
    /// it is dropped.
    pub fn primary_context(&self) -> Result<Context, CudaError> {
        Context::retain_primary(self.api, self.raw)
    }
}
