//! A row-major GEMM on cuBLAS. `C = A B` is `C^T = B^T A^T` in cuBLAS's
//! column-major terms, so B is its first operand and A its second.

use super::sys::{ComputeType, CublasError, DataType, GemmCall, Handle, Operand, Operation};
use crate::problem::{GemmProblem, GemmShape, GemmType};
use ptxc_core::ByteSize;
use ptxc_cuda::DeviceBuffer;
use std::ffi::c_int;
use thiserror::Error;

/// Why a GEMM could not be queued.
#[derive(Debug, Error)]
pub(crate) enum GemmError {
    #[error(transparent)]
    Cublas(#[from] CublasError),
    #[error("{0:?} is too large for cuBLAS's int extents")]
    TooLarge(GemmShape),
    #[error("{operand} holds {actual:?}, but the problem needs {needed:?}")]
    Small { operand: &'static str, needed: ByteSize, actual: ByteSize },
    #[error("C is the same buffer as A or B")]
    Aliased,
    #[error("{0:?} runs only on cuBLASLt")]
    NeedsLt(GemmType),
}

/// cuBLAS's view of a problem: each operand's operation and leading
/// dimension, and its m, n and k.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Mapping {
    pub(super) operations: [Operation; 2],
    pub(super) extents: [c_int; 3],
    pub(super) leading: [c_int; 3],
}

/// Queues `problem` on `handle`'s stream, reading `a` and `b` and writing
/// `c`, a different buffer, each stored as the problem's layout says.
pub(crate) fn gemm(
    handle: &mut Handle<'_>,
    problem: GemmProblem,
    [a, b]: [&DeviceBuffer; 2],
    c: &DeviceBuffer,
) -> Result<(), GemmError> {
    let data = match problem.dtype {
        GemmType::F32 => DataType::F32,
        GemmType::F16 => DataType::F16,
        GemmType::Bf16 => DataType::Bf16,
        narrow @ (GemmType::E4m3 | GemmType::I8) => return Err(GemmError::NeedsLt(narrow)),
    };
    check_buffers(problem, [a, b, c])?;
    let Mapping { operations, extents, leading } = mapping(problem)?;
    let call = GemmCall {
        operations,
        extents,
        a: Operand { address: b.device_address(), leading: leading[0] },
        b: Operand { address: a.device_address(), leading: leading[1] },
        c: Operand { address: c.device_address(), leading: leading[2] },
        data,
        output: data,
        compute: ComputeType::F32,
    };
    Ok(handle.gemm_ex(&call)?)
}

/// cuBLAS's operations, extents and leading dimensions for `problem`.
pub(super) fn mapping(problem: GemmProblem) -> Result<Mapping, GemmError> {
    let int = |extent: u32| c_int::try_from(extent).map_err(|_| GemmError::TooLarge(problem.shape));
    let GemmShape { m, n, k } = problem.shape;
    let (m, n, k) = (int(m)?, int(n)?, int(k)?);
    let layout = problem.layout;
    let (b_operation, b_leading) =
        if layout.b_transposed() { (Operation::Transposed, k) } else { (Operation::AsIs, n) };
    let (a_operation, a_leading) =
        if layout.a_transposed() { (Operation::Transposed, m) } else { (Operation::AsIs, k) };
    // cuBLAS rejects a leading dimension below 1, even for an empty matrix.
    let leading = [b_leading.max(1), a_leading.max(1), n.max(1)];
    Ok(Mapping { operations: [b_operation, a_operation], extents: [n, m, k], leading })
}

/// Fails if `c` is `a` or `b`, or any is smaller than the problem needs.
pub(super) fn check_buffers(
    problem: GemmProblem,
    [a, b, c]: [&DeviceBuffer; 3],
) -> Result<(), GemmError> {
    if [a, b].iter().any(|input| input.device_address() == c.device_address()) {
        return Err(GemmError::Aliased);
    }
    check_sizes(problem, [a, b, c])
}

/// Fails unless `a`, `b` and `c` are at least the problem's sizes.
fn check_sizes(problem: GemmProblem, buffers: [&DeviceBuffer; 3]) -> Result<(), GemmError> {
    let needed = problem.sizes().ok_or(GemmError::TooLarge(problem.shape))?;
    for ((operand, needed), buffer) in ["A", "B", "C"].into_iter().zip(needed).zip(buffers) {
        if buffer.size().get() < needed.get() {
            return Err(GemmError::Small { operand, needed, actual: buffer.size() });
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{GemmError, Mapping, mapping};
    use crate::cublas::sys::Operation::{AsIs, Transposed};
    use crate::problem::{GemmProblem, GemmShape, GemmType, Layout};

    fn problem(layout: Layout, m: u32) -> GemmProblem {
        GemmProblem { shape: GemmShape { m, n: 5, k: 7 }, dtype: GemmType::F16, layout }
    }

    #[test]
    fn each_layout_maps_to_cublas_column_major_operands() {
        let mapped = |layout| mapping(problem(layout, 3)).unwrap();

        assert_eq!(
            mapped(Layout::Nn),
            Mapping { operations: [AsIs, AsIs], extents: [5, 3, 7], leading: [5, 7, 5] }
        );
        assert_eq!(
            mapped(Layout::Nt),
            Mapping { operations: [Transposed, AsIs], extents: [5, 3, 7], leading: [7, 7, 5] }
        );
        assert_eq!(
            mapped(Layout::Tn),
            Mapping { operations: [AsIs, Transposed], extents: [5, 3, 7], leading: [5, 3, 5] }
        );
    }

    #[test]
    fn an_empty_extent_still_has_leading_dimensions_of_one() {
        let mapped = mapping(problem(Layout::Tn, 0)).unwrap();

        assert_eq!(mapped.leading, [5, 1, 5]);
    }

    #[test]
    fn an_extent_past_cublas_int_is_too_large() {
        let error = mapping(problem(Layout::Nn, u32::MAX)).unwrap_err();

        assert!(matches!(error, GemmError::TooLarge(_)), "{error}");
    }
}

#[cfg(all(test, feature = "gpu"))]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod gpu {
    use super::{GemmError, gemm};
    use crate::cublas::sys::Handle;
    use crate::cublas::testing::{context, download, expected, inputs};
    use crate::problem::{GemmProblem, GemmShape, GemmType, Layout};
    use ptxc_core::ByteSize;

    const SHAPE: GemmShape = GemmShape { m: 3, n: 5, k: 7 };

    #[test]
    fn cublas_multiplies_every_type_and_layout_exactly_on_small_integers() {
        let context = context();
        for dtype in [GemmType::F32, GemmType::F16, GemmType::Bf16] {
            for layout in [Layout::Nn, Layout::Nt, Layout::Tn] {
                let problem = GemmProblem { shape: SHAPE, dtype, layout };
                let ([a, b], c) = inputs(&context, problem);
                let stream = context.stream().unwrap();
                let mut handle = Handle::new(&stream).unwrap();

                gemm(&mut handle, problem, [&a, &b], &c).unwrap();
                stream.synchronize().unwrap();

                assert_eq!(download(problem, &c), expected(SHAPE), "{problem:?}");
            }
        }
    }

    #[test]
    fn a_buffer_smaller_than_the_problem_or_aliased_is_refused_before_cublas_runs() {
        let context = context();
        let problem = GemmProblem { shape: SHAPE, dtype: GemmType::F32, layout: Layout::Nn };
        let small = context.alloc(ByteSize::new(4)).unwrap();
        let c = context.alloc(ByteSize::new(60)).unwrap();
        let stream = context.stream().unwrap();
        let mut handle = Handle::new(&stream).unwrap();

        let too_small = gemm(&mut handle, problem, [&small, &small], &c).unwrap_err();
        let aliased = gemm(&mut handle, problem, [&c, &small], &c).unwrap_err();
        let narrow = GemmProblem { dtype: GemmType::E4m3, ..problem };
        let needs_lt = gemm(&mut handle, narrow, [&c, &c], &small).unwrap_err();

        assert!(matches!(too_small, GemmError::Small { operand: "A", .. }), "{too_small}");
        assert!(matches!(aliased, GemmError::Aliased), "{aliased}");
        assert!(matches!(needs_lt, GemmError::NeedsLt(GemmType::E4m3)), "{needs_lt}");
    }
}
