use crate::{NewIndex, sample};
use ptxc_core::{ByteOffset, ByteSize, DType, Dim, Shape};
use ptxc_program::{BufferId, KernelId, Location, Program, ProgramError, Step};

fn rejection(change: impl FnOnce(&mut ptxc_program::ProgramParts)) -> ProgramError {
    let mut parts = sample();
    change(&mut parts);
    Program::new(parts).unwrap_err()
}

#[test]
fn consistent_pieces_form_a_program() {
    let program = Program::new(sample()).unwrap();

    assert_eq!(program.steps().len(), 2);
    assert_eq!(program.constants(), [7; 16]);
}

#[test]
fn a_step_naming_a_missing_kernel_or_buffer_is_rejected() {
    let missing_kernel =
        rejection(|p| p.steps[0] = Step::Launch { kernel: KernelId::new_index(1), args: vec![] });
    let missing_buffer = rejection(|p| {
        p.steps[1] = Step::Copy { from: BufferId::new_index(9), to: BufferId::new_index(3) }
    });

    assert_eq!(missing_kernel, ProgramError::Kernel(KernelId::new_index(1)));
    assert_eq!(missing_buffer, ProgramError::Buffer(BufferId::new_index(9)));
}

#[test]
fn arena_and_constant_buffers_must_be_aligned_and_in_bounds() {
    let id = BufferId::new_index(2);
    let unaligned = rejection(|p| p.buffers[id].location = Location::Arena(ByteOffset::new(8)));
    let past_arena = rejection(|p| p.arena = ByteSize::new(15));
    let past_blob = rejection(|p| p.constants.truncate(12));

    assert_eq!(unaligned, ProgramError::Unaligned(id, ByteOffset::new(8)));
    assert_eq!(past_arena, ProgramError::OutOfBounds(id));
    assert_eq!(past_blob, ProgramError::OutOfBounds(BufferId::new_index(1)));
}

#[test]
fn bindings_must_match_their_buffers() {
    let swapped = rejection(|p| {
        p.inputs[ptxc_program::InputIndex::new_index(0)].buffer = BufferId::new_index(3);
    });
    let reshaped = rejection(|p| {
        p.outputs[ptxc_program::OutputIndex::new_index(0)].shape =
            Shape::new(vec![Dim::new(5)]).unwrap();
    });

    assert_eq!(swapped, ProgramError::Binding(BufferId::new_index(0)));
    assert_eq!(reshaped, ProgramError::Shape(BufferId::new_index(3)));
}

#[test]
fn a_copy_between_different_sizes_is_rejected() {
    let error = rejection(|p| p.buffers[BufferId::new_index(3)].dtype = DType::F16);

    assert_eq!(
        error,
        ProgramError::CopySize { from: BufferId::new_index(2), to: BufferId::new_index(3) }
    );
}
