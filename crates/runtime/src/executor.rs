//! `Executor<S>`: `Eager` launches each step in order on one stream;
//! `Graphed` replays every step as one CUDA graph, recorded once.

use crate::arena::Memory;
use crate::error::RuntimeError;
use crate::io::{download, input_buffers, output_buffers, stage, upload};
use crate::load::LoadedProgram;
use ptxc_core::HostTensor;
use ptxc_cuda::{BlockDim, DeviceBuffer, GraphExec, GridDim, Stream};
use ptxc_program::Step;
use ptxc_target::{BlockCount, ThreadsPerBlock};

/// Launches each step as it comes.
#[derive(Debug)]
pub struct Eager;

/// Replays every step as one CUDA graph, recorded once on fixed buffers for
/// the inputs and outputs, as `TensorRT` and `PyTorch` replay their graphs.
#[derive(Debug)]
pub struct Graphed {
    stream: Stream,
    inputs: Vec<DeviceBuffer>,
    outputs: Vec<DeviceBuffer>,
    graph: GraphExec,
}

/// A program's inputs on the device and its outputs' buffers, ready to run.
#[derive(Debug)]
pub struct Bound<'e> {
    executor: &'e Executor<Eager>,
    inputs: Vec<DeviceBuffer>,
    outputs: Vec<DeviceBuffer>,
}

impl Bound<'_> {
    /// Queues every step on `stream`, a stream of the program's context,
    /// without waiting for them.
    pub fn enqueue(&self, stream: &Stream) -> Result<(), RuntimeError> {
        let loaded = &self.executor.loaded;
        enqueue(loaded, stream, &memory(loaded, &self.inputs, &self.outputs))
    }

    /// The outputs, read back once the queued steps have finished, in the
    /// program's declared order.
    pub fn outputs(&self) -> Result<Vec<HostTensor>, RuntimeError> {
        download(&self.executor.loaded.program, &self.outputs)
    }
}

/// A loaded program and the way it runs.
#[derive(Debug)]
pub struct Executor<S> {
    loaded: LoadedProgram,
    state: S,
}

impl Executor<Eager> {
    /// An executor running `loaded`'s steps eagerly.
    pub const fn eager(loaded: LoadedProgram) -> Self {
        Self { loaded, state: Eager }
    }

    /// The outputs for `inputs`, in the program's declared orders.
    pub fn run(&self, inputs: &[HostTensor]) -> Result<Vec<HostTensor>, RuntimeError> {
        let bound = self.bind(inputs)?;
        let stream = self.loaded.context.stream()?;
        bound.enqueue(&stream)?;
        stream.synchronize()?;
        bound.outputs()
    }

    /// `inputs` on the device, after checking each against its binding,
    /// and room for the outputs, so the program can run repeatedly with
    /// no host copies, as the benchmarks time it.
    pub fn bind(&self, inputs: &[HostTensor]) -> Result<Bound<'_>, RuntimeError> {
        let LoadedProgram { program, context, .. } = &self.loaded;
        let inputs = upload(context, program, inputs)?;
        let outputs = output_buffers(context, program)?;
        Ok(Bound { executor: self, inputs, outputs })
    }

    /// The executor with every step recorded once into a CUDA graph, on
    /// buffers of its own for the inputs and outputs, which each run then
    /// replays with one launch.
    pub fn capture(self) -> Result<Executor<Graphed>, RuntimeError> {
        let LoadedProgram { program, context, .. } = &self.loaded;
        let inputs = input_buffers(context, program)?;
        let outputs = output_buffers(context, program)?;
        let stream = context.stream()?;
        let capture = stream.begin_capture()?;
        enqueue(&self.loaded, capture.stream(), &memory(&self.loaded, &inputs, &outputs))?;
        let graph = capture.end()?.instantiate()?;
        Ok(Executor { loaded: self.loaded, state: Graphed { stream, inputs, outputs, graph } })
    }
}

impl Executor<Graphed> {
    /// The outputs for `inputs`, in the program's declared orders: the
    /// inputs copied into the recorded buffers, the graph replayed, and the
    /// outputs read back.
    pub fn run(&mut self, inputs: &[HostTensor]) -> Result<Vec<HostTensor>, RuntimeError> {
        let Graphed { stream, inputs: staged, outputs, graph } = &mut self.state;
        stage(&self.loaded.program, staged, inputs)?;
        graph.launch(stream)?;
        stream.synchronize()?;
        download(&self.loaded.program, outputs)
    }
}

/// The device memory a run of `loaded` with `inputs` and `outputs` uses.
const fn memory<'m>(
    loaded: &'m LoadedProgram,
    inputs: &'m [DeviceBuffer],
    outputs: &'m [DeviceBuffer],
) -> Memory<'m> {
    Memory { arena: &loaded.arena, constants: &loaded.constants, inputs, outputs }
}

/// Queues each of `loaded`'s steps on `stream`, in order.
fn enqueue(
    loaded: &LoadedProgram,
    stream: &Stream,
    memory: &Memory<'_>,
) -> Result<(), RuntimeError> {
    for step in loaded.program.steps() {
        queue_step(loaded, stream, memory, step)?;
    }
    Ok(())
}

fn queue_step(
    loaded: &LoadedProgram,
    stream: &Stream,
    memory: &Memory<'_>,
    step: &Step,
) -> Result<(), RuntimeError> {
    let program = &loaded.program;
    match step {
        Step::Launch { kernel, args } => {
            let launch = program.kernels()[*kernel].launch;
            let [gx, gy, gz] = launch.grid.map(BlockCount::get);
            let [bx, by, bz] = launch.block.map(ThreadsPerBlock::get);
            let mut call = loaded.functions[*kernel]
                .launch()
                .grid(GridDim::new(gx, gy, gz))
                .block(BlockDim::new(bx, by, bz));
            for &arg in args {
                call = call.arg(memory.slice(program, arg)?);
            }
            Ok(call.launch(stream)?)
        }
        Step::Copy { from, to } => {
            Ok(stream.copy(memory.slice(program, *from)?, memory.slice(program, *to)?)?)
        }
    }
}
