//! Assembles the scheduled kernels, a buffer for every tensor they touch
//! and the steps into a `Program`.
//!
//! Graph inputs, and outputs a node produces, are bound to the caller's
//! memory before any group runs, so stages write outputs in place. Other
//! values and each group's intermediates get their own arena offset, and
//! constants and tables their own place in the blob.
// NOTE: one arena allocation per buffer, never reused, is the trivial plan
// M2 calls for; lifetime-based planners come with M4.

use crate::compilation::Scheduled;
use crate::error::CompileError;
use ptxc_codegen::generate;
use ptxc_core::{ByteOffset, ByteSize, DType, HostTensor, Idx, IndexVec, Shape};
use ptxc_graph::{Graph, ValueDef, ValueId};
use ptxc_loop::{Access, LoweredKernel};
use ptxc_lower::{Binding, LoweredGroup};
use ptxc_program::{
    BUFFER_ALIGNMENT, Buffer, BufferId, InputIndex, IoBinding, KernelId, KernelSpec, LaunchConfig,
    Location, OutputIndex, Program, ProgramParts, Step,
};
use ptxc_target::Target;
use std::collections::HashMap;

/// The program for `scheduled`'s kernels.
pub(crate) fn emit(scheduled: &Scheduled, target: &Target) -> Result<Program, CompileError> {
    let graph = scheduled.partitioned.graph();
    let mut emitter = Emitter::new(graph);
    emitter.bind_inputs()?;
    let copies = emitter.bind_outputs()?;
    for (group, kernels) in scheduled.groups.iter().zip(&scheduled.kernels) {
        emitter.group(group, kernels, target)?;
    }
    for (value, output) in copies {
        let from = emitter.value_buffer(value)?;
        emitter.steps.push(Step::Copy { from, to: output });
    }
    Ok(Program::new(emitter.finish(target))?)
}

struct Emitter<'g> {
    graph: &'g Graph,
    values: HashMap<ValueId, BufferId>,
    buffers: IndexVec<BufferId, Buffer>,
    constants: Vec<u8>,
    arena: ByteOffset,
    kernels: IndexVec<KernelId, KernelSpec>,
    steps: Vec<Step>,
    inputs: IndexVec<InputIndex, IoBinding>,
    outputs: IndexVec<OutputIndex, IoBinding>,
}

impl<'g> Emitter<'g> {
    fn new(graph: &'g Graph) -> Self {
        Self {
            graph,
            values: HashMap::new(),
            buffers: IndexVec::new(),
            constants: Vec::new(),
            arena: ByteOffset::new(0),
            kernels: IndexVec::new(),
            steps: Vec::new(),
            inputs: IndexVec::new(),
            outputs: IndexVec::new(),
        }
    }

    fn finish(self, target: &Target) -> ProgramParts {
        ProgramParts {
            target: target.sm(),
            kernels: self.kernels,
            constants: self.constants,
            arena: ByteSize::new(self.arena.get()),
            buffers: self.buffers,
            steps: self.steps,
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }

    fn bind_inputs(&mut self) -> Result<(), CompileError> {
        for &value in self.graph.inputs() {
            let binding = self.io_binding(value, self.buffers.next_id())?;
            let location = Location::Input(self.inputs.push(binding));
            let buffer = self.buffer_of(value, location);
            let _new = self.values.insert(value, buffer);
        }
        Ok(())
    }

    /// Binds every output, in the graph's order. An output a node produces,
    /// the first time it appears, is where the node writes; any other
    /// output, an input, a constant or a repeat, is returned with the value
    /// to copy into it once the groups have run.
    fn bind_outputs(&mut self) -> Result<Vec<(ValueId, BufferId)>, CompileError> {
        let mut copies = Vec::new();
        for &value in self.graph.outputs() {
            let produced = matches!(self.graph.value(value).def(), ValueDef::Output { .. });
            let id = self.output(value)?;
            if produced && !self.values.contains_key(&value) {
                let _new = self.values.insert(value, id);
            } else {
                copies.push((value, id));
            }
        }
        Ok(copies)
    }

    /// A new output buffer for `value`, bound as the next output.
    fn output(&mut self, value: ValueId) -> Result<BufferId, CompileError> {
        let binding = self.io_binding(value, self.buffers.next_id())?;
        let location = Location::Output(self.outputs.push(binding));
        Ok(self.buffer_of(value, location))
    }

    fn io_binding(&self, value: ValueId, buffer: BufferId) -> Result<IoBinding, CompileError> {
        let found = self.graph.value(value);
        let name = found.name().ok_or(CompileError::Unnamed(value))?.as_str().to_owned();
        Ok(IoBinding { name, shape: found.ty().shape().clone(), buffer })
    }

    /// A buffer of `value`'s type at `location`.
    fn buffer_of(&mut self, value: ValueId, location: Location) -> BufferId {
        let ty = self.graph.value(value).ty();
        self.buffers.push(Buffer {
            dtype: ty.dtype(),
            elements: ty.shape().element_count(),
            location,
        })
    }

    /// `value`'s buffer, placing it on first use: a constant in the blob,
    /// anything else in the arena.
    fn value_buffer(&mut self, value: ValueId) -> Result<BufferId, CompileError> {
        if let Some(&id) = self.values.get(&value) {
            return Ok(id);
        }
        let id = match self.graph.value(value).def() {
            ValueDef::Constant(constant) => self.constant(self.graph.constant(constant))?,
            ValueDef::Input(_) | ValueDef::Output { .. } => {
                let ty = self.graph.value(value).ty();
                self.arena_buffer(ty.dtype(), ty.shape())?
            }
        };
        let _new = self.values.insert(value, id);
        Ok(id)
    }

    /// A buffer for `tensor` at the next aligned offset of the blob.
    fn constant(&mut self, tensor: &HostTensor) -> Result<BufferId, CompileError> {
        let length = u64::try_from(self.constants.len()).map_err(|_| CompileError::TooLarge)?;
        let offset = ByteOffset::new(length).align_up(BUFFER_ALIGNMENT)?;
        let start = usize::try_from(offset.get()).map_err(|_| CompileError::TooLarge)?;
        self.constants.resize(start, 0);
        self.constants.extend(tensor.data().to_le_bytes());
        let elements = tensor.shape().element_count();
        let location = Location::Constant(offset);
        Ok(self.buffers.push(Buffer { dtype: tensor.dtype(), elements, location }))
    }

    /// A buffer of `dtype` and `shape` at the next aligned arena offset.
    fn arena_buffer(&mut self, dtype: DType, shape: &Shape) -> Result<BufferId, CompileError> {
        let elements = shape.element_count();
        let offset = self.arena.align_up(BUFFER_ALIGNMENT)?;
        let end = offset.get().checked_add(elements.in_bytes(dtype)?.get());
        self.arena = ByteOffset::new(end.ok_or(CompileError::TooLarge)?);
        Ok(self.buffers.push(Buffer { dtype, elements, location: Location::Arena(offset) }))
    }

    /// Each of `group`'s stages as a kernel and a launch.
    fn group(
        &mut self,
        group: &LoweredGroup,
        kernels: &[LoweredKernel],
        target: &Target,
    ) -> Result<(), CompileError> {
        let tables = group
            .tables()
            .iter()
            .map(|table| self.constant(table))
            .collect::<Result<Vec<_>, _>>()?;
        let intermediates = group
            .intermediates()
            .iter()
            .map(|ty| self.arena_buffer(ty.dtype(), ty.shape()))
            .collect::<Result<Vec<_>, _>>()?;
        for (stage, kernel) in group.stages().iter().zip(kernels) {
            let mut reads = stage.inputs().iter();
            let mut writes = stage.outputs().iter();
            let mut args = Vec::new();
            for param in kernel.params() {
                let binding = match param.access {
                    Access::Read => reads.next(),
                    Access::Write => writes.next(),
                };
                let binding = *binding.ok_or(CompileError::StageArity)?;
                let buffer = match binding {
                    Binding::Value(value) => Some(self.value_buffer(value)?),
                    Binding::Intermediate(id) => intermediates.get(id.index()).copied(),
                    Binding::Table(id) => tables.get(id.index()).copied(),
                };
                args.push(buffer.ok_or(CompileError::StageArity)?);
            }
            let launch = LaunchConfig { grid: kernel.launch().grid, block: kernel.launch().block };
            let ptx = generate(kernel, target)?.to_string();
            let spec = KernelSpec { name: kernel.name().clone(), ptx, launch };
            self.steps.push(Step::Launch { kernel: self.kernels.push(spec), args });
        }
        Ok(())
    }
}
