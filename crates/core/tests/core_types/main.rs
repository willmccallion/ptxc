//! Core vocabulary types: typed indices, shapes and axes, sizes, host
//! tensors and quantization parameters.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    reason = "test code: a panic is a test failure, and scales are compared exactly on purpose"
)]

mod index;
mod quant;
mod shape;
mod size;
mod tensor;
