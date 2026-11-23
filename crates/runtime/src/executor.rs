//! `Executor<S>`: `Eager` launches each step in order on one stream. A
//! graphed executor replaying a captured CUDA graph comes with M4.

use crate::arena::Memory;
use crate::error::RuntimeError;
use crate::io::{download, output_buffers, upload};
use crate::load::LoadedProgram;
use ptxc_core::HostTensor;
use ptxc_cuda::{BlockDim, DeviceBuffer, GridDim, Stream};
use ptxc_program::Step;
use ptxc_target::{BlockCount, ThreadsPerBlock};
use std::marker::PhantomData;

/// Launches each step as it comes.
#[derive(Debug)]
pub struct Eager;

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
        let LoadedProgram { program, arena, constants, .. } = &self.executor.loaded;
        let memory = Memory { arena, constants, inputs: &self.inputs, outputs: &self.outputs };
        for step in program.steps() {
            self.executor.step(stream, &memory, step)?;
        }
        Ok(())
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
    mode: PhantomData<S>,
}

impl Executor<Eager> {
    /// An executor running `loaded`'s steps eagerly.
    pub const fn eager(loaded: LoadedProgram) -> Self {
        Self { loaded, mode: PhantomData }
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

    fn step(&self, stream: &Stream, memory: &Memory<'_>, step: &Step) -> Result<(), RuntimeError> {
        let program = &self.loaded.program;
        match step {
            Step::Launch { kernel, args } => {
                let launch = program.kernels()[*kernel].launch;
                let [gx, gy, gz] = launch.grid.map(BlockCount::get);
                let [bx, by, bz] = launch.block.map(ThreadsPerBlock::get);
                let mut call = self.loaded.functions[*kernel]
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
}
