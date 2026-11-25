//! Heuristic choices for workloads that have no tuning record: a matrix
//! product, or a convolution as an implicit GEMM, is tiled on tensor cores
//! where the target has an `mma` for its types and on CUDA cores
//! otherwise, an injective stage spreads over blocks of threads, and any
//! other reduction gives each output row a block whose threads combine
//! through shared memory.

use crate::rules::conv_implicit_gemm::{gemm_loops, implicit_gemm};
use crate::rules::matmul_loops::{GemmLoops, product_loops};
use crate::rules::{elementwise, matmul_mma, matmul_shape, matmul_simt, reduction};
use ptxc_loop::{ComputeDef, IterKind, Open, Schedule, ScheduleError, Validated};
use ptxc_target::Target;
use std::sync::Arc;

/// The schedule `compute` gets with no tuning record, validated for
/// `target`.
pub fn default_schedule(
    compute: Arc<ComputeDef>,
    target: &Target,
) -> Result<Schedule<Validated>, ScheduleError> {
    let reduces = compute.iter_vars().iter().any(|axis| axis.kind == IterKind::Reduce);
    let product = matmul_shape::product(&compute);
    let convolution = implicit_gemm(&compute);
    let mut schedule = Schedule::new(compute);
    let loops = match (product, convolution) {
        (Some(product), _) => Some(product_loops(&schedule, product)?),
        (None, Some(convolution)) => Some(gemm_loops(&mut schedule, &convolution)?),
        (None, None) => None,
    };
    match loops {
        Some(loops) => tiled(&mut schedule, &loops, target)?,
        None if reduces => reduction::one_block_per_row(&mut schedule)?,
        None => elementwise::flatten_over_blocks(&mut schedule)?,
    }
    schedule.validate(target)
}

/// A matrix product's `loops` tiled on tensor cores where `target` has an
/// `mma` for its operands' and sum's types, and on CUDA cores otherwise.
fn tiled(
    schedule: &mut Schedule<Open>,
    loops: &GemmLoops,
    target: &Target,
) -> Result<(), ScheduleError> {
    match matmul_mma::instruction(schedule.compute(), loops.operands, target) {
        Some(instruction) => {
            matmul_mma::tiled_on_tensor_cores(schedule, loops, (instruction, target))
        }
        None => matmul_simt::tiled_on_cuda_cores(schedule, loops, target),
    }
}
