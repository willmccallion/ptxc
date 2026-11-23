//! Test infrastructure: tolerance comparison, golden files, ONNX test-case loading, ptxas checks, random graph generation and shrinking.

pub mod compare;
pub mod data;
mod generate;
pub mod golden;
pub mod onnx_tests;
pub mod ptxas;
pub mod reference;
mod shrink;
