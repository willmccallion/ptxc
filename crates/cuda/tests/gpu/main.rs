//! The driver bindings on a real GPU. Compiled only with `--features gpu`;
//! without a GPU these fail rather than skip.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod device;
mod graph;
mod launch;
mod memory;
mod module;

use ptxc_cuda::{Context, DeviceOrdinal, Driver};

/// Device 0's primary context.
pub(crate) fn context() -> Context {
    Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap().primary_context()
}

/// The hand-written PTX for `vector_add(a, b, out, n)` on f32.
pub(crate) const VECTOR_ADD: &str = include_str!("../fixtures/vector_add.ptx");

/// The hand-written PTX for `spin(cycles)`, which busy-waits.
pub(crate) const SPIN: &str = include_str!("../fixtures/spin.ptx");

/// `values` as their native-endian bytes.
pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_ne_bytes()).collect()
}

/// Native-endian bytes back as f32s.
pub(crate) fn from_f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|chunk| f32::from_ne_bytes(*chunk)).collect()
}
