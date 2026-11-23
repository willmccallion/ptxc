//! The versioned on-disk format (D21), checked once on load: the magic
//! `PTXC`, the format version and the header's length as little-endian
//! integers, a JSON header, then the raw constant blob.

use crate::buffer_plan::{Buffer, BufferId, InputIndex, Location, OutputIndex};
use crate::kernel::{KernelId, KernelSpec, LaunchConfig};
use crate::program::{IoBinding, Program, ProgramError, ProgramParts};
use crate::step::Step;
use ptxc_core::{ByteOffset, ByteSize, DType, Dim, ElementCount, Idx, KernelName, Shape};
use ptxc_target::{BlockCount, SmVersion, ThreadsPerBlock};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The first bytes of every program file.
const MAGIC: [u8; 4] = *b"PTXC";

/// The format this crate writes and reads.
pub const FORMAT_VERSION: u32 = 1;

/// The magic, the version and the header length.
const PREAMBLE_BYTES: usize = 16;

/// Bytes that are not a program this crate can read.
#[derive(Debug, Error)]
pub enum FormatError {
    /// The file does not start with `PTXC`.
    #[error("not a ptxc program: the file does not start with PTXC")]
    Magic,
    /// The file is from another format version.
    #[error("program format {0}, but this ptxc reads format {FORMAT_VERSION}; recompile the model")]
    Version(u32),
    /// The file ends before its header does.
    #[error("the program file is truncated")]
    Truncated,
    /// The header is too long to record.
    #[error("the program header is too large to save")]
    TooLarge,
    /// The header is not valid JSON of the expected shape.
    #[error("malformed program header: {0}")]
    Header(#[from] serde_json::Error),
    /// The header names an unknown element type.
    #[error("unknown element type `{0}`")]
    DType(String),
    /// The header gives an invalid shape.
    #[error("invalid shape {0:?}")]
    Shape(Vec<u64>),
    /// The pieces do not form a program.
    #[error(transparent)]
    Program(#[from] ProgramError),
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    target: [u8; 2],
    arena_bytes: u64,
    kernels: Vec<KernelEntry>,
    buffers: Vec<BufferEntry>,
    steps: Vec<StepEntry>,
    inputs: Vec<IoEntry>,
    outputs: Vec<IoEntry>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct KernelEntry {
    name: String,
    ptx: String,
    grid: [u32; 3],
    block: [u32; 3],
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BufferEntry {
    dtype: String,
    elements: u64,
    location: LocationEntry,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum LocationEntry {
    Arena(u64),
    Constant(u64),
    Input(usize),
    Output(usize),
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum StepEntry {
    Launch { kernel: usize, args: Vec<usize> },
    Copy { from: usize, to: usize },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IoEntry {
    name: String,
    dims: Vec<u64>,
    buffer: usize,
}

/// `program` as the bytes of a program file.
pub fn save(program: &Program) -> Result<Vec<u8>, FormatError> {
    let header = serde_json::to_vec(&header_of(program))?;
    let length = u64::try_from(header.len()).map_err(|_| FormatError::TooLarge)?;
    let mut bytes = Vec::with_capacity(PREAMBLE_BYTES + header.len() + program.constants().len());
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(program.constants());
    Ok(bytes)
}

/// The program a program file holds.
pub fn load(bytes: &[u8]) -> Result<Program, FormatError> {
    let (magic, rest) = bytes.split_first_chunk::<4>().ok_or(FormatError::Truncated)?;
    if *magic != MAGIC {
        return Err(FormatError::Magic);
    }
    let (version, rest) = rest.split_first_chunk::<4>().ok_or(FormatError::Truncated)?;
    let version = u32::from_le_bytes(*version);
    if version != FORMAT_VERSION {
        return Err(FormatError::Version(version));
    }
    let (length, rest) = rest.split_first_chunk::<8>().ok_or(FormatError::Truncated)?;
    let length =
        usize::try_from(u64::from_le_bytes(*length)).map_err(|_| FormatError::Truncated)?;
    let (header, constants) = rest.split_at_checked(length).ok_or(FormatError::Truncated)?;
    let header: Header = serde_json::from_slice(header)?;
    Ok(Program::new(parts_of(header, constants.to_vec())?)?)
}

fn header_of(program: &Program) -> Header {
    Header {
        target: [program.target().major(), program.target().minor()],
        arena_bytes: program.arena().get(),
        kernels: program.kernels().iter().map(kernel_entry).collect(),
        buffers: program.buffers().iter().map(buffer_entry).collect(),
        steps: program.steps().iter().map(step_entry).collect(),
        inputs: program.inputs().iter().map(io_entry).collect(),
        outputs: program.outputs().iter().map(io_entry).collect(),
    }
}

fn kernel_entry(kernel: &KernelSpec) -> KernelEntry {
    KernelEntry {
        name: kernel.name.as_str().to_owned(),
        ptx: kernel.ptx.clone(),
        grid: kernel.launch.grid.map(BlockCount::get),
        block: kernel.launch.block.map(ThreadsPerBlock::get),
    }
}

fn buffer_entry(buffer: &Buffer) -> BufferEntry {
    let location = match buffer.location {
        Location::Arena(offset) => LocationEntry::Arena(offset.get()),
        Location::Constant(offset) => LocationEntry::Constant(offset.get()),
        Location::Input(index) => LocationEntry::Input(index.index()),
        Location::Output(index) => LocationEntry::Output(index.index()),
    };
    BufferEntry { dtype: buffer.dtype.name().to_owned(), elements: buffer.elements.get(), location }
}

fn step_entry(step: &Step) -> StepEntry {
    match step {
        Step::Launch { kernel, args } => StepEntry::Launch {
            kernel: kernel.index(),
            args: args.iter().map(|a| a.index()).collect(),
        },
        Step::Copy { from, to } => StepEntry::Copy { from: from.index(), to: to.index() },
    }
}

fn io_entry(binding: &IoBinding) -> IoEntry {
    IoEntry {
        name: binding.name.clone(),
        dims: binding.shape.dims().iter().map(|dim| dim.get()).collect(),
        buffer: binding.buffer.index(),
    }
}

fn parts_of(header: Header, constants: Vec<u8>) -> Result<ProgramParts, FormatError> {
    let [major, minor] = header.target;
    Ok(ProgramParts {
        target: SmVersion::new(major, minor),
        kernels: header.kernels.into_iter().map(kernel_spec).collect(),
        constants,
        arena: ByteSize::new(header.arena_bytes),
        buffers: header.buffers.into_iter().map(buffer).collect::<Result<_, _>>()?,
        steps: header.steps.into_iter().map(step).collect(),
        inputs: header.inputs.into_iter().map(io_binding).collect::<Result<_, _>>()?,
        outputs: header.outputs.into_iter().map(io_binding).collect::<Result<_, _>>()?,
    })
}

fn kernel_spec(entry: KernelEntry) -> KernelSpec {
    let launch = LaunchConfig {
        grid: entry.grid.map(BlockCount::new),
        block: entry.block.map(ThreadsPerBlock::new),
    };
    KernelSpec { name: KernelName::new(entry.name), ptx: entry.ptx, launch }
}

fn buffer(entry: BufferEntry) -> Result<Buffer, FormatError> {
    let dtype = DType::from_name(&entry.dtype).ok_or(FormatError::DType(entry.dtype))?;
    let location = match entry.location {
        LocationEntry::Arena(offset) => Location::Arena(ByteOffset::new(offset)),
        LocationEntry::Constant(offset) => Location::Constant(ByteOffset::new(offset)),
        LocationEntry::Input(index) => Location::Input(InputIndex::new(index)),
        LocationEntry::Output(index) => Location::Output(OutputIndex::new(index)),
    };
    Ok(Buffer { dtype, elements: ElementCount::new(entry.elements), location })
}

fn step(entry: StepEntry) -> Step {
    match entry {
        StepEntry::Launch { kernel, args } => Step::Launch {
            kernel: KernelId::new(kernel),
            args: args.into_iter().map(BufferId::new).collect(),
        },
        StepEntry::Copy { from, to } => {
            Step::Copy { from: BufferId::new(from), to: BufferId::new(to) }
        }
    }
}

fn io_binding(entry: IoEntry) -> Result<IoBinding, FormatError> {
    let shape = Shape::new(entry.dims.iter().copied().map(Dim::new).collect())
        .map_err(|_| FormatError::Shape(entry.dims))?;
    Ok(IoBinding { name: entry.name, shape, buffer: BufferId::new(entry.buffer) })
}
