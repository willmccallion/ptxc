//! Unrolled loops expanded before selection, so a scratch array they index
//! takes constant offsets, which ptxas keeps in registers.

use crate::target;
use ptxc_codegen::generate;
use ptxc_core::{ByteSize, DType, Dim, Shape};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeName, Primitive, ReduceOp, Schedule, SplitFactor, TensorExpr,
    TensorName, ThreadAxis, VarName,
};
use std::sync::Arc;

/// The stack frame of a 256 by 256 by 256 f32 matmul scheduled as the SIMT
/// rule schedules one: the block rows and columns fused onto `blockIdx.x`,
/// blocks of 16 by 16 threads each accumulating a 4 by 4 tile, and the
/// reduction in steps of 8 with both operands staged in shared memory; the
/// step's depth and the tile unrolled if `unrolled`.
fn stack_frame(unrolled: bool) -> ByteSize {
    let shape = |dims: [u64; 2]| Shape::new(dims.map(Dim::new).to_vec()).unwrap();
    let mut compute = ComputeBuilder::new(ComputeName::new("matmul"));
    let lhs = compute.input(TensorName::new("a"), DType::F32, shape([256, 256]));
    let rhs = compute.input(TensorName::new("b"), DType::F32, shape([256, 256]));
    let (_, row) = compute.spatial(VarName::new("i"), Dim::new(256));
    let (_, column) = compute.spatial(VarName::new("j"), Dim::new(256));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(256));
    let load = |tensor, index| Box::new(TensorExpr::Load { tensor, index });
    let body =
        TensorExpr::Binary(BinaryOp::Mul, load(lhs, vec![row, k]), load(rhs, vec![k, column]));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body: Box::new(body) };
    compute.output(TensorName::new("out"), sum);
    let mut schedule = Schedule::new(Arc::new(compute.finish().unwrap()));
    let nest = schedule.loops().nest().to_vec();
    let factors = |values: &[u64]| values.iter().map(|&f| SplitFactor::new(f).unwrap()).collect();
    let rows = schedule.split(nest[0], factors(&[16, 4])).unwrap();
    let columns = schedule.split(nest[1], factors(&[16, 4])).unwrap();
    let steps = schedule.split(nest[2], factors(&[8])).unwrap();
    let order =
        vec![rows[0], columns[0], rows[1], columns[1], steps[0], steps[1], rows[2], columns[2]];
    schedule.apply(Primitive::Reorder { order }).unwrap();
    let grid = schedule.fuse(vec![rows[0], columns[0]]).unwrap();
    let binds = [
        (grid, ThreadAxis::BlockX),
        (rows[1], ThreadAxis::ThreadY),
        (columns[1], ThreadAxis::ThreadX),
    ];
    for (loop_, axis) in binds {
        schedule.apply(Primitive::Bind { loop_, axis }).unwrap();
    }
    for tensor in [lhs, rhs] {
        schedule.apply(Primitive::CacheRead { tensor, at: steps[0] }).unwrap();
    }
    if unrolled {
        for loop_ in [steps[1], rows[2], columns[2]] {
            schedule.apply(Primitive::Unroll { loop_ }).unwrap();
        }
    }
    let kernel = schedule.validate(&target()).unwrap().lower();
    let module = generate(&kernel, &target()).unwrap();
    ptxc_testkit::ptxas::stack_frame(&module.to_string(), &target().sm().to_string()).unwrap()
}

#[test]
fn an_accumulator_indexed_by_an_unrolled_loop_needs_no_stack_frame() {
    assert_eq!(stack_frame(true), ByteSize::new(0));
    assert!(stack_frame(false).get() > 0);
}
