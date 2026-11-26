use crate::{add, compile, graph};
use ptxc_compiler::{ArenaError, ArenaUse, arena_use, verify_arena};
use ptxc_core::{ByteOffset, ByteSize, DType, ElementCount, Idx, IndexVec, KernelName};
use ptxc_program::{
    Buffer, BufferId, KernelId, KernelSpec, LaunchConfig, Location, Program, ProgramParts, Step,
};
use ptxc_target::{BlockCount, SmVersion, ThreadsPerBlock};

/// A program of arena buffers, each 16 f32s at its offset, run by `steps`,
/// in an arena of `arena` bytes.
fn arena_program(offsets: &[u64], steps: Vec<Step>, arena: u64) -> Program {
    let buffers: IndexVec<BufferId, Buffer> = offsets
        .iter()
        .map(|&offset| Buffer {
            dtype: DType::F32,
            elements: ElementCount::new(16),
            location: Location::Arena(ByteOffset::new(offset)),
        })
        .collect();
    let launch =
        LaunchConfig { grid: [BlockCount::new(1); 3], block: [ThreadsPerBlock::new(1); 3] };
    let kernel = KernelSpec { name: KernelName::new("k"), ptx: String::new(), launch };
    Program::new(ProgramParts {
        target: SmVersion::SM_89,
        kernels: std::iter::once(kernel).collect(),
        constants: Vec::new(),
        arena: ByteSize::new(arena),
        buffers,
        steps,
        inputs: IndexVec::new(),
        outputs: IndexVec::new(),
    })
    .unwrap()
}

fn launch(args: &[usize]) -> Step {
    Step::Launch {
        kernel: KernelId::new(0),
        args: args.iter().map(|&a| BufferId::new(a)).collect(),
    }
}

#[test]
fn buffers_live_at_the_same_step_may_not_share_bytes() {
    let program = arena_program(&[0, 0], vec![launch(&[0]), launch(&[0, 1])], 64);

    assert_eq!(
        verify_arena(&program),
        Err(ArenaError::Overlap { first: BufferId::new(0), second: BufferId::new(1) })
    );
}

#[test]
fn buffers_live_at_different_steps_may_share_bytes() {
    let steps = vec![launch(&[0]), launch(&[0, 1]), launch(&[1, 2])];
    let program = arena_program(&[0, 256, 0], steps, 320);

    assert_eq!(verify_arena(&program), Ok(()));
    assert_eq!(
        arena_use(&program),
        Ok(ArenaUse { size: ByteSize::new(320), lower_bound: ByteSize::new(128) })
    );
}

#[test]
fn a_copy_keeps_its_source_live_until_it_runs() {
    let copy = Step::Copy { from: BufferId::new(0), to: BufferId::new(2) };
    let steps = vec![launch(&[0]), launch(&[1]), copy];
    let program = arena_program(&[0, 0, 256], steps, 320);

    assert_eq!(
        verify_arena(&program),
        Err(ArenaError::Overlap { first: BufferId::new(0), second: BufferId::new(1) })
    );
}

#[test]
fn a_chain_reuses_the_bytes_of_values_it_no_longer_reads() {
    let program = compile(graph(|b, x, y| {
        let first = add(b, x, y);
        let second = add(b, first, x);
        let third = add(b, second, y);
        vec![add(b, third, x)]
    }))
    .unwrap();

    let written: Vec<Location> = program
        .steps()
        .iter()
        .filter_map(|step| match step {
            Step::Launch { args, .. } => Some(program.buffers()[args[2]].location),
            Step::Copy { .. } => None,
        })
        .collect();
    let arena = |offset| Location::Arena(ByteOffset::new(offset));
    assert_eq!(written[..3], [arena(0), arena(256), arena(0)]);
    assert_eq!(verify_arena(&program), Ok(()));
    assert_eq!(
        arena_use(&program),
        Ok(ArenaUse { size: ByteSize::new(276), lower_bound: ByteSize::new(40) })
    );
}
