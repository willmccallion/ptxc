//! Core vocabulary types: typed indices, shapes and axes, sizes, host
//! tensors, quantization parameters, kernel names and durations.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    reason = "test code: a panic is a test failure, and scales are compared exactly on purpose"
)]

mod dtype;
mod index;
mod kernel_name;
mod quant;
mod shape;
mod size;
mod tensor;
mod time;
