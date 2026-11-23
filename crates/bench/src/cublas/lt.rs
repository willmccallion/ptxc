//! GEMMs on cuBLASLt: every type, FP8 and int8 included, and the top
//! algorithms its heuristic proposes, from which the sweep takes the best
//! as an upper bound beside the default (BENCHMARKING.md).

use super::gemm::{GemmError, Mapping, check_buffers, mapping};
use super::sys::Candidate;
use super::sys::{
    ComputeType, CublasError, DataType, LtHandle, LtProblem, MatrixLayout, Operation, Plan,
};
use crate::problem::{GemmProblem, GemmType};
use ptxc_core::ByteSize;
use ptxc_cuda::DeviceBuffer;
use std::ffi::c_int;

/// How many of the heuristic's algorithms the upper bound tries.
pub(crate) const TOP: u8 = 8;

/// The most workspace an algorithm may use: 32 MiB.
pub(crate) const MAX_WORKSPACE: ByteSize = ByteSize::new(32 << 20);

/// `problem`'s plan: up to `TOP` algorithms, best first, or none if
/// cuBLASLt cannot run it.
pub(crate) fn plan(handle: &LtHandle<'_>, problem: GemmProblem) -> Result<Plan, GemmError> {
    let Mapping { operations, extents: [m, n, k], leading } = mapping(problem)?;
    let (data, output, compute) = types(problem.dtype);
    let [b_operation, a_operation] = operations;
    let layouts = [
        stored(b_operation, [m, k], leading[0])?,
        stored(a_operation, [k, n], leading[1])?,
        stored(Operation::AsIs, [m, n], leading[2])?,
    ];
    let lt = LtProblem { operations, layouts, data, output, compute };
    Ok(handle.plan(&lt, TOP, MAX_WORKSPACE)?)
}

/// Queues `problem` with `candidate`, one of `plan`'s, reading `a` and `b`
/// and writing `c`, a different buffer, with `workspace`.
pub(crate) fn run(
    handle: &mut LtHandle<'_>,
    (plan, candidate): (&Plan, &Candidate),
    problem: GemmProblem,
    [a, b, c]: [&DeviceBuffer; 3],
    workspace: &DeviceBuffer,
) -> Result<(), GemmError> {
    check_buffers(problem, [a, b, c])?;
    let operands = [b.device_address(), a.device_address(), c.device_address()];
    Ok(handle.matmul(plan, candidate, operands, workspace)?)
}

/// A and B's type, C's type and the compute type for `dtype`.
const fn types(dtype: GemmType) -> (DataType, DataType, ComputeType) {
    match dtype {
        GemmType::F32 => (DataType::F32, DataType::F32, ComputeType::F32),
        GemmType::F16 => (DataType::F16, DataType::F16, ComputeType::F32),
        GemmType::Bf16 => (DataType::Bf16, DataType::Bf16, ComputeType::F32),
        GemmType::E4m3 => (DataType::E4m3, DataType::F32, ComputeType::F32),
        GemmType::I8 => (DataType::I8, DataType::I32, ComputeType::I32),
    }
}

/// The layout of an operand whose operation gives `[rows, columns]`: as
/// stored, before the operation.
fn stored(
    operation: Operation,
    [rows, columns]: [c_int; 2],
    leading: c_int,
) -> Result<MatrixLayout, GemmError> {
    let (rows, columns) = match operation {
        Operation::AsIs => (rows, columns),
        Operation::Transposed => (columns, rows),
    };
    let extent = |value: c_int| u64::try_from(value).map_err(|_| CublasError::Size(0));
    Ok(MatrixLayout { rows: extent(rows)?, columns: extent(columns)?, leading: i64::from(leading) })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{stored, types};
    use crate::cublas::sys::{ComputeType, DataType, MatrixLayout, Operation};
    use crate::problem::GemmType;

    #[test]
    fn a_transposed_operand_is_described_as_stored() {
        let as_is = stored(Operation::AsIs, [5, 7], 5).unwrap();
        let transposed = stored(Operation::Transposed, [5, 7], 7).unwrap();

        assert_eq!(as_is, MatrixLayout { rows: 5, columns: 7, leading: 5 });
        assert_eq!(transposed, MatrixLayout { rows: 7, columns: 5, leading: 7 });
    }

    #[test]
    fn narrow_types_write_and_compute_wider() {
        assert_eq!(types(GemmType::E4m3), (DataType::E4m3, DataType::F32, ComputeType::F32));
        assert_eq!(types(GemmType::I8), (DataType::I8, DataType::I32, ComputeType::I32));
        assert_eq!(types(GemmType::Bf16), (DataType::Bf16, DataType::Bf16, ComputeType::F32));
    }
}

#[cfg(all(test, feature = "gpu"))]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod gpu {
    use super::{TOP, plan, run};
    use crate::cublas::sys::LtHandle;
    use crate::cublas::testing::{context, download, expected, inputs};
    use crate::problem::{GemmProblem, GemmShape, GemmType, Layout};
    use ptxc_core::ByteSize;

    const TYPES: [GemmType; 5] =
        [GemmType::F32, GemmType::F16, GemmType::Bf16, GemmType::E4m3, GemmType::I8];
    const LAYOUTS: [Layout; 3] = [Layout::Nn, Layout::Nt, Layout::Tn];

    /// What cuBLASLt proposes no algorithm for on `sm_89`: FP8 runs only
    /// with A transposed and B as is in its column-major terms, which is
    /// the row-major NT.
    const UNSUPPORTED: [(GemmType, Layout); 2] =
        [(GemmType::E4m3, Layout::Nn), (GemmType::E4m3, Layout::Tn)];

    #[test]
    fn every_algorithm_cublaslt_proposes_computes_exactly() {
        let context = context();
        let shape = GemmShape { m: 32, n: 16, k: 32 };
        let mut unsupported = Vec::new();
        for dtype in TYPES {
            for layout in LAYOUTS {
                let problem = GemmProblem { shape, dtype, layout };
                let stream = context.stream().unwrap();
                let mut handle = LtHandle::new(&stream).unwrap();

                let plan = plan(&handle, problem).unwrap();

                if plan.candidates.is_empty() {
                    unsupported.push((dtype, layout));
                    continue;
                }
                assert!(plan.candidates.len() <= usize::from(TOP), "{problem:?}");
                let ([a, b], c) = inputs(&context, problem);
                let most = plan.candidates.iter().map(|candidate| candidate.workspace.get()).max();
                let workspace = context.alloc(ByteSize::new(most.unwrap().max(1))).unwrap();
                for candidate in &plan.candidates {
                    run(&mut handle, (&plan, candidate), problem, [&a, &b, &c], &workspace)
                        .unwrap();
                    stream.synchronize().unwrap();
                    assert_eq!(download(problem, &c), expected(shape), "{problem:?} {candidate:?}");
                }
            }
        }
        assert_eq!(unsupported, UNSUPPORTED);
    }
}
