//! Test infrastructure: tolerance comparison, golden files, ONNX test-case loading, ptxas checks, op-family case generators, running lowered graphs, random graph generation and shrinking.

pub mod compare;
pub mod data;
pub mod families;
mod generate;
pub mod golden;
pub mod lowered;
pub mod onnx_tests;
pub mod ptxas;
pub mod reference;
mod shrink;
