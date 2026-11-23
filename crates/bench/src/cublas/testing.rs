//! Data for the GPU tests: small integers, exact in every type, stored as
//! a problem's layout says, and their product on the host.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

use crate::host::{data, values};
use crate::problem::{GemmProblem, GemmShape, GemmType};
use ptxc_core::convert::{int_to_f64, truncate};
use ptxc_core::{ByteSize, TensorData};
use ptxc_cuda::{Context, DeviceBuffer, DeviceOrdinal, Driver};

/// Device 0's primary context.
pub(super) fn context() -> Context {
    Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap().primary_context()
}

/// An entry of a matrix, an integer in [-2, 2].
fn entry(row: u32, column: u32, stride: u32) -> i64 {
    i64::from((row * stride + column) % 5) - 2
}

/// The `rows` by `columns` matrix of `entry`, row-major, stored
/// transposed if `transposed`.
fn stored(rows: u32, columns: u32, stride: u32, transposed: bool) -> Vec<i64> {
    let (outer, inner) = if transposed { (columns, rows) } else { (rows, columns) };
    (0..outer)
        .flat_map(|o| {
            (0..inner)
                .map(move |i| if transposed { entry(i, o, stride) } else { entry(o, i, stride) })
        })
        .collect()
}

fn encode(dtype: GemmType, values: &[i64]) -> Vec<u8> {
    let floats: Vec<f64> = values.iter().map(|&v| int_to_f64(v)).collect();
    data(dtype.dtype(), &floats).unwrap().to_le_bytes()
}

fn upload(context: &Context, bytes: &[u8]) -> DeviceBuffer {
    let mut buffer = context.alloc(ByteSize::new(u64::try_from(bytes.len()).unwrap())).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer
}

/// A and B for `problem` on the device, and room for C.
pub(super) fn inputs(context: &Context, problem: GemmProblem) -> ([DeviceBuffer; 2], DeviceBuffer) {
    let GemmShape { m, n, k } = problem.shape;
    let a = stored(m, k, 7, problem.layout.a_transposed());
    let b = stored(k, n, 3, problem.layout.b_transposed());
    let [_, _, c_bytes] = problem.sizes().unwrap();
    let a = upload(context, &encode(problem.dtype, &a));
    let b = upload(context, &encode(problem.dtype, &b));
    ([a, b], context.alloc(c_bytes).unwrap())
}

/// C, read back from the device as integers.
pub(super) fn download(problem: GemmProblem, c: &DeviceBuffer) -> Vec<i64> {
    let mut bytes = vec![0; usize::try_from(c.size().get()).unwrap()];
    c.copy_to_host(&mut bytes).unwrap();
    let output = TensorData::from_le_bytes(problem.dtype.output(), &bytes).unwrap();
    values(&output)
        .unwrap()
        .into_iter()
        .map(|value| {
            assert!(value.fract() == 0.0 && value.abs() < 1e6, "{value}");
            truncate(value)
        })
        .collect()
}

/// `C = A B` on the host, row-major.
pub(super) fn expected(shape: GemmShape) -> Vec<i64> {
    let GemmShape { m, n, k } = shape;
    (0..m)
        .flat_map(|i| (0..n).map(move |j| (0..k).map(|p| entry(i, p, 7) * entry(p, j, 3)).sum()))
        .collect()
}
