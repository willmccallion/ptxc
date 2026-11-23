//! Kernels built for crate-internal tests.

use crate::compute::{
    BinaryOp, ComputeBuilder, ComputeName, Scalar, TensorExpr, TensorName, VarName,
};
use crate::schedule::{Primitive, Schedule, SplitFactor, ThreadAxis};
use crate::stmt::LoweredKernel;
use ptxc_core::{DType, Dim, Shape};
use ptxc_target::{SmVersion, preset};
use std::sync::Arc;

/// `y[i] = x[i] + 1` over 8 elements, the sum split by 4 with the inner
/// loop bound to `threadIdx.x`: buffers 0 (`x`) and 1 (`y`).
pub(crate) fn kernel() -> LoweredKernel {
    kernel_with(None)
}

/// As `kernel`, with the outer loop bound to `blockIdx.x`, so two blocks
/// of four threads run.
pub(crate) fn two_block_kernel() -> LoweredKernel {
    kernel_with(Some(ThreadAxis::BlockX))
}

fn kernel_with(outer: Option<ThreadAxis>) -> LoweredKernel {
    let mut compute = ComputeBuilder::new(ComputeName::new("inc"));
    let x = compute.input(TensorName::new("x"), DType::F32, Shape::new(vec![Dim::new(8)]).unwrap());
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(8));
    let one = TensorExpr::Const(Scalar::float(DType::F32, 1.0).unwrap());
    let load = TensorExpr::Load { tensor: x, index: vec![i] };
    compute.output(
        TensorName::new("y"),
        TensorExpr::Binary(BinaryOp::Add, Box::new(load), Box::new(one)),
    );
    let mut schedule = Schedule::new(Arc::new(compute.finish().unwrap()));
    let nest = schedule.loops().nest().to_vec();
    let parts = schedule.split(nest[0], vec![SplitFactor::new(4).unwrap()]).unwrap();
    schedule.apply(Primitive::Bind { loop_: parts[1], axis: ThreadAxis::ThreadX }).unwrap();
    if let Some(axis) = outer {
        schedule.apply(Primitive::Bind { loop_: parts[0], axis }).unwrap();
    }
    schedule.validate(&preset(SmVersion::SM_89).unwrap()).unwrap().lower()
}
