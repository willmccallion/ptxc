//! Swizzled shared tiles: a constant index takes its swizzled offset, and a
//! computed one is swizzled with a shift, a mask and an exclusive-or.

use crate::ptxas::assemble;
use crate::target;
use ptxc_codegen::generate;
use ptxc_core::{DType, Dim, Idx, Shape};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeName, Primitive, ReduceOp, Schedule, SplitFactor, Swizzle,
    TensorExpr, TensorName, TensorParamId, VarName,
};
use std::sync::Arc;

/// The PTX of an f32 product of 8 by 8 and 8 by 4 on one thread, both
/// operands staged whole and `a`'s tile swizzled, every loop but the stage
/// loop unrolled, so every tile read is at a constant index.
fn single_thread_product() -> String {
    let shape = |dims: &[u64]| Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap();
    let mut compute = ComputeBuilder::new(ComputeName::new("product"));
    let lhs = compute.input(TensorName::new("a"), DType::F32, shape(&[8, 8]));
    let rhs = compute.input(TensorName::new("b"), DType::F32, shape(&[8, 4]));
    let (_, row) = compute.spatial(VarName::new("i"), Dim::new(8));
    let (_, column) = compute.spatial(VarName::new("j"), Dim::new(4));
    let (axis, step) = compute.reduce_axis(VarName::new("k"), Dim::new(8));
    let load = |tensor, index| Box::new(TensorExpr::Load { tensor, index });
    let (left, right) = (load(lhs, vec![row, step]), load(rhs, vec![step, column]));
    let body = Box::new(TensorExpr::Binary(BinaryOp::Mul, left, right));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![axis], body };
    compute.output(TensorName::new("out"), sum);
    let mut schedule = Schedule::new(Arc::new(compute.finish().unwrap()));
    let nest = schedule.loops().nest().to_vec();
    let depth = schedule.split(nest[2], vec![SplitFactor::new(8).unwrap()]).unwrap();
    schedule
        .apply(Primitive::Reorder { order: vec![depth[0], nest[0], nest[1], depth[1]] })
        .unwrap();
    for tensor in [0, 1].map(TensorParamId::new) {
        schedule.apply(Primitive::CacheRead { tensor, at: depth[0] }).unwrap();
    }
    let swizzle = Swizzle::for_rows(32).unwrap();
    schedule.apply(Primitive::Swizzle { tensor: TensorParamId::new(0), swizzle }).unwrap();
    for loop_ in [nest[0], nest[1], depth[1]] {
        schedule.apply(Primitive::Unroll { loop_ }).unwrap();
    }
    let module = generate(&schedule.validate(&target()).unwrap().lower(), &target()).unwrap();
    assemble(&module);
    module.to_string()
}

#[test]
fn a_constant_index_into_a_swizzled_tile_reads_at_its_swizzled_offset() {
    let module = single_thread_product();

    // The first read of each element of `a`'s tile, in the order the
    // unrolled body first reads them: row by row.
    let mut offsets: Vec<u64> = Vec::new();
    for line in module.lines().filter(|line| line.contains("ld.shared.f32")) {
        let Some(rest) = line.split("[shared_0").nth(1) else { continue };
        let offset = rest.trim_start_matches('+').split(']').next().unwrap();
        let offset = if offset.is_empty() { 0 } else { offset.parse().unwrap() };
        if !offsets.contains(&offset) {
            offsets.push(offset);
        }
    }

    let swizzle = Swizzle::for_rows(32).unwrap();
    let expected: Vec<u64> =
        (0..8).flat_map(|row| (0..8).map(move |k| swizzle.apply(row * 32 + k * 4))).collect();
    assert!(module.contains(".shared .align 16 .b8 shared_0[256];"), "{module}");
    assert_eq!(offsets, expected);
}
