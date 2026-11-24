//! Heuristic choices for workloads that have no tuning record: a matrix
//! product is tiled on tensor cores where the target has an `mma` for its
//! types and on CUDA cores otherwise, an injective stage spreads over
//! blocks of threads, and any other reduction gives each output row a
//! block whose threads combine through shared memory.

use crate::rules::{elementwise, matmul_mma, matmul_shape, matmul_simt, reduction};
use ptxc_loop::{ComputeDef, IterKind, Schedule, ScheduleError, Validated};
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
    let instruction =
        product.and_then(|product| matmul_mma::instruction(&compute, product, target));
    let mut schedule = Schedule::new(compute);
    match (product, instruction) {
        (Some(product), Some(instruction)) => {
            matmul_mma::tiled_on_tensor_cores(&mut schedule, product, instruction)?;
        }
        (Some(product), None) => matmul_simt::tiled_on_cuda_cores(&mut schedule, product)?,
        (None, _) if reduces => reduction::one_block_per_row(&mut schedule)?,
        (None, _) => elementwise::flatten_over_blocks(&mut schedule)?,
    }
    schedule.validate(target)
}
