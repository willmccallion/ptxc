//! `Executor<S>`: `Eager` launches each step in order on one stream. A
//! graphed executor replaying a captured CUDA graph comes with M4.

use crate::arena::Memory;
use crate::error::RuntimeError;
use crate::io::{download, output_buffers, upload};
use crate::load::LoadedProgram;
use ptxc_core::HostTensor;
use ptxc_cuda::{BlockDim, GridDim, Stream};
use ptxc_program::Step;
use ptxc_target::{BlockCount, ThreadsPerBlock};
use std::marker::PhantomData;

/// Launches each step as it comes.
#[derive(Debug)]
pub struct Eager;

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
        let LoadedProgram { program, context, arena, constants, .. } = &self.loaded;
        let inputs = upload(context, program, inputs)?;
        let outputs = output_buffers(context, program)?;
        let memory = Memory { arena, constants, inputs: &inputs, outputs: &outputs };
        let stream = context.stream()?;
        for step in program.steps() {
            self.step(&stream, &memory, step)?;
        }
        stream.synchronize()?;
        download(program, &outputs)
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
