use crate::{add, compile, constant, f32s, graph};
use ptxc_compiler::CompileError;
use ptxc_core::{ByteOffset, ByteSize, Idx};
use ptxc_graph::GraphBuilder;
use ptxc_program::{BufferId, InputIndex, Location, OutputIndex, Step, load, save};

fn location(program: &ptxc_program::Program, buffer: BufferId) -> Location {
    program.buffers()[buffer].location
}

#[test]
fn a_node_becomes_one_launch_over_its_inputs_and_output() {
    let program = compile(graph(|b, x, y| vec![add(b, x, y)])).unwrap();

    let [Step::Launch { args, .. }] = program.steps() else { panic!("{:?}", program.steps()) };
    let locations: Vec<Location> = args.iter().map(|&arg| location(&program, arg)).collect();
    assert_eq!(
        locations,
        [
            Location::Input(InputIndex::new(0)),
            Location::Input(InputIndex::new(1)),
            Location::Output(OutputIndex::new(0))
        ]
    );
    assert_eq!(program.inputs()[InputIndex::new(1)].name, "y");
    assert_eq!(program.outputs()[OutputIndex::new(0)].name, "out0");
    assert_eq!(program.arena(), ByteSize::new(0));
}

#[test]
fn a_value_between_nodes_lives_in_the_arena() {
    let program = compile(graph(|b, x, y| {
        let sum = add(b, x, y);
        vec![add(b, sum, x)]
    }))
    .unwrap();

    let [Step::Launch { args: first, .. }, Step::Launch { args: second, .. }] = program.steps()
    else {
        panic!("{:?}", program.steps());
    };
    assert_eq!(location(&program, first[2]), Location::Arena(ByteOffset::new(0)));
    assert_eq!(second[0], first[2]);
    assert_eq!(program.arena(), ByteSize::new(20));
}

#[test]
fn constants_are_placed_aligned_in_the_blob() {
    let program = compile(graph(|b, x, _| {
        let (one, two) = (constant(b, 1.0), constant(b, 2.0));
        let sum = add(b, x, one);
        vec![add(b, sum, two)]
    }))
    .unwrap();

    let constants: Vec<Location> = program
        .buffers()
        .iter()
        .map(|buffer| buffer.location)
        .filter(|location| matches!(location, Location::Constant(_)))
        .collect();
    assert_eq!(
        constants,
        [Location::Constant(ByteOffset::new(0)), Location::Constant(ByteOffset::new(256))]
    );
    assert_eq!(program.constants().len(), 256 + 20);
    assert_eq!(&program.constants()[256..260], 2.0_f32.to_le_bytes());
}

#[test]
fn an_output_that_is_an_input_or_a_repeat_is_copied() {
    let program = compile(graph(|b, x, y| {
        let sum = add(b, x, y);
        vec![x, sum, sum]
    }))
    .unwrap();

    let copies: Vec<(Location, Location)> = program
        .steps()
        .iter()
        .filter_map(|step| match *step {
            Step::Copy { from, to } => Some((location(&program, from), location(&program, to))),
            Step::Launch { .. } => None,
        })
        .collect();
    assert_eq!(
        copies,
        [
            (Location::Input(InputIndex::new(0)), Location::Output(OutputIndex::new(0))),
            (Location::Output(OutputIndex::new(1)), Location::Output(OutputIndex::new(2))),
        ]
    );
}

#[test]
fn every_emitted_kernel_assembles_and_the_program_round_trips() {
    let program = compile(graph(|b, x, y| {
        let sum = add(b, x, y);
        vec![add(b, sum, x)]
    }))
    .unwrap();

    for kernel in program.kernels() {
        ptxc_testkit::ptxas::assemble(&kernel.ptx, "sm_89").unwrap();
    }
    assert_eq!(load(&save(&program).unwrap()).unwrap(), program);
}

#[test]
fn an_unnamed_input_is_rejected() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32s(5));
    let y = add(&mut builder, x, x);
    builder.set_name(y, ptxc_graph::ValueName::new("y".into()));
    let graph = builder.finish(vec![y]).unwrap();

    assert!(matches!(compile(graph), Err(CompileError::Unnamed(value)) if value == x));
}
