//! Test infrastructure: tolerance comparison, golden files, ONNX test-case loading, ptxas checks, op-family case generators, running lowered graphs, random graph generation and shrinking, and tensorized products.

pub mod compare;
pub mod data;
pub mod families;
pub mod generate;
pub mod golden;
pub mod lowered;
pub mod onnx_tests;
pub mod pipelined;
pub mod ptxas;
pub mod reference;
pub mod shrink;
pub mod sums;
pub mod tensorized;
