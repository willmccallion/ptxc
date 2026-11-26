//! The testkit through its public API.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod bounded;
mod compare;
mod convolution;
mod data;
mod generate;
mod golden;
mod onnx_tests;
mod ptxas;
mod reference;
mod shrink;
mod sums;
mod tensorized;
