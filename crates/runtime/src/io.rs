//! Binding host inputs and outputs: each checked against the program's
//! bindings, staged into its own device allocation, and read back.

use crate::error::RuntimeError;
use crate::load::allocation;
use ptxc_core::{HostTensor, TensorData};
use ptxc_cuda::{Context, DeviceBuffer};
use ptxc_program::{IoBinding, Program};

/// `inputs` on the device, after checking each is the type and shape its
/// binding declares.
pub(crate) fn upload(
    context: &Context,
    program: &Program,
    inputs: &[HostTensor],
) -> Result<Vec<DeviceBuffer>, RuntimeError> {
    let mut buffers = input_buffers(context, program)?;
    stage(program, &mut buffers, inputs)?;
    Ok(buffers)
}

/// Copies `inputs` into `buffers`, one per input, after checking each is
/// the type and shape its binding declares.
pub(crate) fn stage(
    program: &Program,
    buffers: &mut [DeviceBuffer],
    inputs: &[HostTensor],
) -> Result<(), RuntimeError> {
    let expected = program.inputs().len();
    if inputs.len() != expected {
        return Err(RuntimeError::InputCount { expected, actual: inputs.len() });
    }
    for ((binding, input), buffer) in program.inputs().iter().zip(inputs).zip(buffers) {
        let dtype = program.buffers()[binding.buffer].dtype;
        if input.dtype() != dtype || input.shape() != &binding.shape {
            let name = binding.name.clone();
            return Err(RuntimeError::Input { name, dtype, shape: binding.shape.clone() });
        }
        let bytes = input.data().to_le_bytes();
        if !bytes.is_empty() {
            buffer.copy_from_host(&bytes)?;
        }
    }
    Ok(())
}

/// A device allocation for each output.
pub(crate) fn output_buffers(
    context: &Context,
    program: &Program,
) -> Result<Vec<DeviceBuffer>, RuntimeError> {
    io_buffers(context, program, program.outputs().iter())
}

/// A device allocation for each input.
pub(crate) fn input_buffers(
    context: &Context,
    program: &Program,
) -> Result<Vec<DeviceBuffer>, RuntimeError> {
    io_buffers(context, program, program.inputs().iter())
}

/// A device allocation for each of `bindings`, as large as its buffer.
fn io_buffers<'p>(
    context: &Context,
    program: &Program,
    bindings: impl Iterator<Item = &'p IoBinding>,
) -> Result<Vec<DeviceBuffer>, RuntimeError> {
    bindings
        .map(|binding| {
            let buffer = program.buffers()[binding.buffer];
            let size =
                buffer.elements.in_bytes(buffer.dtype).map_err(|_| RuntimeError::TooLarge)?;
            allocation(context, size)
        })
        .collect()
}

/// Each output read back, with its binding's shape.
pub(crate) fn download(
    program: &Program,
    outputs: &[DeviceBuffer],
) -> Result<Vec<HostTensor>, RuntimeError> {
    program
        .outputs()
        .iter()
        .zip(outputs)
        .map(|(binding, device)| {
            let buffer = program.buffers()[binding.buffer];
            let size =
                buffer.elements.in_bytes(buffer.dtype).map_err(|_| RuntimeError::TooLarge)?;
            let mut bytes =
                vec![0_u8; usize::try_from(size.get()).map_err(|_| RuntimeError::TooLarge)?];
            if !bytes.is_empty() {
                device.copy_to_host(&mut bytes)?;
            }
            let data = TensorData::from_le_bytes(buffer.dtype, &bytes)?;
            Ok(HostTensor::new(binding.shape.clone(), data)?)
        })
        .collect()
}
