//! Runs a lowered kernel on device 0 and on the loop interpreter.

use ptxc_codegen::generate;
use ptxc_core::{ByteSize, HostTensor, TensorData};
use ptxc_cuda::{BlockDim, Context, Device, DeviceBuffer, DeviceOrdinal, Driver, GridDim};
use ptxc_loop::{Access, ComputeDef, LoweredKernel, RunError};
use ptxc_target::{BlockCount, SmVersion, Target, ThreadsPerBlock, preset};
use ptxc_testkit::compare::{Tolerance, compare};
use std::sync::{Arc, OnceLock};

fn target() -> Target {
    preset(SmVersion::SM_89).unwrap()
}

/// Device 0, opened once, so its context lives for the whole test run
/// rather than being made again for every case.
pub(crate) fn device() -> &'static Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE.get_or_init(|| Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap())
}

fn context() -> Context {
    device().primary_context()
}

/// `compute` under its default schedule.
pub(crate) fn lowered(compute: ComputeDef) -> LoweredKernel {
    ptxc_schedule::default_schedule(Arc::new(compute), &target()).unwrap().lower()
}

/// `kernel`'s outputs on the GPU for `inputs`, ordered as the loop
/// interpreter takes and returns them.
pub(crate) fn run_on_device(kernel: &LoweredKernel, inputs: &[HostTensor]) -> Vec<HostTensor> {
    let context = context();
    let module = context.load_ptx(&generate(kernel, &target()).unwrap().to_string()).unwrap();
    let function = module.function(kernel.name()).unwrap();
    let mut remaining_inputs = inputs.iter();
    let buffers: Vec<(Access, DeviceBuffer)> = kernel
        .params()
        .iter()
        .map(|param| {
            let buffer = &kernel.buffers()[param.buffer];
            let bytes = buffer.size.in_bytes(buffer.dtype).unwrap();
            // NOTE: the driver refuses zero-byte allocations; an empty
            // tensor gets one byte it never touches.
            let mut device = context.alloc(ByteSize::new(bytes.get().max(1))).unwrap();
            if param.access == Access::Read {
                let bytes = remaining_inputs.next().unwrap().data().to_le_bytes();
                if !bytes.is_empty() {
                    device.copy_from_host(&bytes).unwrap();
                }
            }
            (param.access, device)
        })
        .collect();
    let launch = kernel.launch();
    let [gx, gy, gz] = launch.grid.map(BlockCount::get);
    let [bx, by, bz] = launch.block.map(ThreadsPerBlock::get);
    let stream = context.stream().unwrap();
    let mut call =
        function.launch().grid(GridDim::new(gx, gy, gz)).block(BlockDim::new(bx, by, bz));
    for (_, device) in &buffers {
        call = call.arg(device);
    }
    call.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    kernel
        .params()
        .iter()
        .zip(&buffers)
        .filter(|(param, _)| param.access == Access::Write)
        .map(|(param, (_, device))| {
            let buffer = &kernel.buffers()[param.buffer];
            let bytes = buffer.size.in_bytes(buffer.dtype).unwrap();
            let mut host = vec![0_u8; usize::try_from(bytes.get()).unwrap()];
            if !host.is_empty() {
                device.copy_to_host(&mut host).unwrap();
            }
            HostTensor::vector(TensorData::from_le_bytes(buffer.dtype, &host).unwrap())
        })
        .collect()
}

/// Runs `compute` on the GPU and on the loop interpreter, and checks every
/// output matches exactly.
pub(crate) fn check_against_interpreter(compute: ComputeDef, inputs: &[HostTensor]) {
    let kernel = lowered(compute);
    let _outputs = match_interpreter(&kernel, inputs, "").unwrap();
}

/// `kernel`'s outputs from the loop interpreter, after checking the GPU's
/// match them exactly; `context` describes the case if they do not. An
/// input the interpreter rejects leaves the GPU's result undefined, so it
/// is returned as the interpreter's error, unchecked.
pub(crate) fn match_interpreter(
    kernel: &LoweredKernel,
    inputs: &[HostTensor],
    context: &str,
) -> Result<Vec<HostTensor>, RunError> {
    let expected = ptxc_loop::run(kernel, inputs)?;
    let actual = run_on_device(kernel, inputs);
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        if let Err(mismatch) = compare(actual, expected, Tolerance::Exact) {
            panic!(
                "{}: output {index} differs from the interpreter:\n{mismatch}\n{context}",
                kernel.name()
            );
        }
    }
    Ok(expected)
}
