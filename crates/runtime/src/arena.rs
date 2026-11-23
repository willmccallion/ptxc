//! Where each of a program's buffers is on the device: a slice of the
//! arena, of the constants, or of an input's or output's own allocation.

use crate::error::RuntimeError;
use ptxc_core::{ByteOffset, Idx};
use ptxc_cuda::{DeviceBuffer, DeviceSlice};
use ptxc_program::{BufferId, Location, Program};

/// The device memory one run of a program uses.
pub(crate) struct Memory<'a> {
    pub(crate) arena: &'a DeviceBuffer,
    pub(crate) constants: &'a DeviceBuffer,
    pub(crate) inputs: &'a [DeviceBuffer],
    pub(crate) outputs: &'a [DeviceBuffer],
}

impl<'a> Memory<'a> {
    /// `buffer`'s bytes on the device.
    pub(crate) fn slice(
        &self,
        program: &Program,
        buffer: BufferId,
    ) -> Result<DeviceSlice<'a>, RuntimeError> {
        let found = program.buffers()[buffer];
        let size = found.elements.in_bytes(found.dtype).map_err(|_| RuntimeError::TooLarge)?;
        let (allocation, offset) = match found.location {
            Location::Arena(offset) => (self.arena, offset),
            Location::Constant(offset) => (self.constants, offset),
            Location::Input(index) => (&self.inputs[index.index()], ByteOffset::new(0)),
            Location::Output(index) => (&self.outputs[index.index()], ByteOffset::new(0)),
        };
        Ok(allocation.slice(offset, size)?)
    }
}
