//! Programs through their public API: the checks on their pieces, and the
//! file format.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod checks;
mod format;

use ptxc_core::{ByteOffset, ByteSize, DType, Dim, ElementCount, IndexVec, KernelName, Shape};
use ptxc_program::{
    Buffer, BufferId, InputIndex, IoBinding, KernelId, KernelSpec, LaunchConfig, Location,
    OutputIndex, ProgramParts, Step,
};
use ptxc_target::{BlockCount, SmVersion, ThreadsPerBlock};

/// `y = k(x, w)` through an intermediate `t`, then copied out: an input, a
/// constant, an arena buffer and an output.
pub(crate) fn sample() -> ProgramParts {
    let f32s = |location| Buffer { dtype: DType::F32, elements: ElementCount::new(4), location };
    let buffers: IndexVec<BufferId, Buffer> = [
        f32s(Location::Input(InputIndex::new_index(0))),
        f32s(Location::Constant(ByteOffset::new(0))),
        f32s(Location::Arena(ByteOffset::new(0))),
        f32s(Location::Output(OutputIndex::new_index(0))),
    ]
    .into_iter()
    .collect();
    let launch = LaunchConfig {
        grid: [BlockCount::new(1); 3],
        block: [ThreadsPerBlock::new(4), ThreadsPerBlock::new(1), ThreadsPerBlock::new(1)],
    };
    let kernel = KernelSpec { name: KernelName::new("k"), ptx: ".version 8.4\n".into(), launch };
    let shape = Shape::new(vec![Dim::new(2), Dim::new(2)]).unwrap();
    let binding = |name: &str, buffer| IoBinding {
        name: name.into(),
        shape: shape.clone(),
        buffer: BufferId::new_index(buffer),
    };
    ProgramParts {
        target: SmVersion::SM_89,
        kernels: std::iter::once(kernel).collect(),
        constants: vec![7; 16],
        arena: ByteSize::new(16),
        buffers,
        steps: vec![
            Step::Launch {
                kernel: KernelId::new_index(0),
                args: (0..3).map(BufferId::new_index).collect(),
            },
            Step::Copy { from: BufferId::new_index(2), to: BufferId::new_index(3) },
        ],
        inputs: std::iter::once(binding("x", 0)).collect(),
        outputs: std::iter::once(binding("y", 3)).collect(),
    }
}

/// `Idx::new` without importing the trait at every call.
trait NewIndex: ptxc_core::Idx {
    fn new_index(index: usize) -> Self {
        Self::new(index)
    }
}

impl<T: ptxc_core::Idx> NewIndex for T {}
