//! Test infrastructure: tolerance comparison, golden files, ONNX test-case loading, random graph generation and shrinking.

pub mod compare;
pub mod data;
mod generate;
pub mod golden;
mod onnx_tests;
mod shrink;
