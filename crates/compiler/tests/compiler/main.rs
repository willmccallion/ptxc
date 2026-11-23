//! Graphs compiled to programs: the buffers, steps and kernels emitted.
//! Whether those kernels compute the right values is the GPU tests' job.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod emit;

use ptxc_compiler::{Compilation, CompileError};
use ptxc_core::{DType, Dim, HostTensor, Shape};
use ptxc_graph::{BinaryOp, Graph, GraphBuilder, Layout, Op, TensorType, ValueId, ValueName};
use ptxc_program::Program;
use ptxc_target::{SmVersion, preset};

pub(crate) fn f32s(length: u64) -> TensorType {
    TensorType::new(DType::F32, Shape::new(vec![Dim::new(length)]).unwrap(), Layout::RowMajor)
}

/// A graph from `build`, which adds nodes to a builder holding inputs
/// `x` and `y`, both f32 vectors of 5, and returns the outputs.
pub(crate) fn graph(
    build: impl FnOnce(&mut GraphBuilder, ValueId, ValueId) -> Vec<ValueId>,
) -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32s(5));
    let y = builder.add_input(f32s(5));
    builder.set_name(x, ValueName::new("x".into()));
    builder.set_name(y, ValueName::new("y".into()));
    let outputs = build(&mut builder, x, y);
    for (n, &output) in outputs.iter().enumerate() {
        builder.set_name(output, ValueName::new(format!("out{n}")));
    }
    builder.finish(outputs).unwrap()
}

pub(crate) fn add(builder: &mut GraphBuilder, a: ValueId, b: ValueId) -> ValueId {
    builder.add_node(Op::Binary(BinaryOp::Add), vec![a, b]).unwrap()[0]
}

pub(crate) fn constant(builder: &mut GraphBuilder, value: f32) -> ValueId {
    builder.add_constant(HostTensor::vector(vec![value; 5].into()))
}

pub(crate) fn compile(graph: Graph) -> Result<Program, CompileError> {
    let target = preset(SmVersion::SM_89).unwrap();
    Compilation::from_graph(graph, target).optimize()?.partition()?.lower()?.schedule()?.emit()
}
