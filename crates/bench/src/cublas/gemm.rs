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
}

/// cuBLAS's view of a problem: each operand's operation and leading
/// dimension, and its m, n and k.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Mapping {
    operations: [Operation; 2],
    extents: [c_int; 3],
    leading: [c_int; 3],
}

/// Queues `problem` on `handle`'s stream, reading `a` and `b` and writing
/// `c`, a different buffer, each stored as the problem's layout says.
pub(crate) fn gemm(
    handle: &mut Handle<'_>,
    problem: GemmProblem,
    [a, b]: [&DeviceBuffer; 2],
    c: &DeviceBuffer,
) -> Result<(), GemmError> {
    if [a, b].iter().any(|input| input.device_address() == c.device_address()) {
        return Err(GemmError::Aliased);
    }
    check_sizes(problem, [a, b, c])?;
    let Mapping { operations, extents, leading } = mapping(problem)?;
    let data = data_type(problem.dtype);
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
fn mapping(problem: GemmProblem) -> Result<Mapping, GemmError> {
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

const fn data_type(dtype: GemmType) -> DataType {
    match dtype {
        GemmType::F32 => DataType::F32,
        GemmType::F16 => DataType::F16,
        GemmType::Bf16 => DataType::Bf16,
    }
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
#[allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]
mod gpu {
    use super::{GemmError, gemm};
    use crate::cublas::sys::Handle;
    use crate::problem::{GemmProblem, GemmShape, GemmType, Layout};
    use ptxc_core::convert::{f64_to_f32, int_to_f64};
    use ptxc_core::float::{Bf16, F16};
    use ptxc_core::{ByteSize, TensorData};
    use ptxc_cuda::{Context, DeviceBuffer, DeviceOrdinal, Driver};

    const SHAPE: GemmShape = GemmShape { m: 3, n: 5, k: 7 };

    /// Small integers, so every product and sum is exact in every type.
    fn entry(row: u32, column: u32, stride: u32) -> f64 {
        int_to_f64(i64::from((row * stride + column) % 5)) - 2.0
    }

    /// The `rows` by `columns` matrix of `entry`, row-major, stored
    /// transposed if `transposed`.
    fn stored(rows: u32, columns: u32, stride: u32, transposed: bool) -> Vec<f64> {
        let (outer, inner) = if transposed { (columns, rows) } else { (rows, columns) };
        (0..outer)
            .flat_map(|o| {
                (0..inner).map(
                    move |i| if transposed { entry(i, o, stride) } else { entry(o, i, stride) },
                )
            })
            .collect()
    }

    fn encode(dtype: GemmType, values: &[f64]) -> Vec<u8> {
        let data: TensorData = match dtype {
            GemmType::F32 => values.iter().map(|&v| f64_to_f32(v)).collect::<Vec<_>>().into(),
            GemmType::F16 => values.iter().map(|&v| F16::from_f64(v)).collect::<Vec<_>>().into(),
            GemmType::Bf16 => values.iter().map(|&v| Bf16::from_f64(v)).collect::<Vec<_>>().into(),
        };
        data.to_le_bytes()
    }

    fn decode(dtype: GemmType, bytes: &[u8]) -> Vec<f64> {
        match TensorData::from_le_bytes(dtype.dtype(), bytes).unwrap() {
            TensorData::F32(v) => v.into_iter().map(f64::from).collect(),
            TensorData::F16(v) => v.into_iter().map(F16::to_f64).collect(),
            TensorData::Bf16(v) => v.into_iter().map(Bf16::to_f64).collect(),
            other => panic!("{:?}", other.dtype()),
        }
    }

    fn upload(context: &Context, bytes: &[u8]) -> DeviceBuffer {
        let mut buffer = context.alloc(ByteSize::new(u64::try_from(bytes.len()).unwrap())).unwrap();
        buffer.copy_from_host(bytes).unwrap();
        buffer
    }

    /// `C = A B` on the host, row-major.
    fn expected() -> Vec<f64> {
        let GemmShape { m, n, k } = SHAPE;
        (0..m)
            .flat_map(|i| {
                (0..n).map(move |j| (0..k).map(|p| entry(i, p, 7) * entry(p, j, 3)).sum())
            })
            .collect()
    }

    fn run(context: &Context, problem: GemmProblem) -> Vec<f64> {
        let GemmShape { m, n, k } = problem.shape;
        let a = upload(
            context,
            &encode(problem.dtype, &stored(m, k, 7, problem.layout.a_transposed())),
        );
        let b = upload(
            context,
            &encode(problem.dtype, &stored(k, n, 3, problem.layout.b_transposed())),
        );
        let [_, _, c_bytes] = problem.sizes().unwrap();
        let c = context.alloc(c_bytes).unwrap();
        let stream = context.stream().unwrap();
        let mut handle = Handle::new(&stream).unwrap();

        gemm(&mut handle, problem, [&a, &b], &c).unwrap();
        stream.synchronize().unwrap();

        let mut bytes = vec![0; usize::try_from(c_bytes.get()).unwrap()];
        c.copy_to_host(&mut bytes).unwrap();
        decode(problem.dtype, &bytes)
    }

    #[test]
    fn cublas_multiplies_every_type_and_layout_exactly_on_small_integers() {
        let context =
            Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap().primary_context();
        for dtype in [GemmType::F32, GemmType::F16, GemmType::Bf16] {
            for layout in [Layout::Nn, Layout::Nt, Layout::Tn] {
                let problem = GemmProblem { shape: SHAPE, dtype, layout };

                let product = run(&context, problem);

                assert_eq!(product, expected(), "{problem:?}");
            }
        }
    }

    #[test]
    fn a_buffer_smaller_than_the_problem_is_refused_before_cublas_runs() {
        let context =
            Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap().primary_context();
        let problem = GemmProblem { shape: SHAPE, dtype: GemmType::F32, layout: Layout::Nn };
        let small = context.alloc(ByteSize::new(4)).unwrap();
        let c = context.alloc(ByteSize::new(60)).unwrap();
        let stream = context.stream().unwrap();
        let mut handle = Handle::new(&stream).unwrap();

        let too_small = gemm(&mut handle, problem, [&small, &small], &c).unwrap_err();
        let aliased = gemm(&mut handle, problem, [&c, &small], &c).unwrap_err();

        assert!(matches!(too_small, GemmError::Small { operand: "A", .. }), "{too_small}");
        assert!(matches!(aliased, GemmError::Aliased), "{aliased}");
    }
}
