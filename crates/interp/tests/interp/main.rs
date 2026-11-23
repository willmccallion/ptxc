//! The interpreter through its public API, on graphs written in the text
//! format, against values worked out by hand or with NumPy from ONNX's
//! definitions.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod activation;
mod attention;
mod builders;
mod conv;
mod elementwise;
mod gather;
mod interpreter;
mod matmul;
mod norm;
mod pool;
mod quant;
mod reduce;
mod resize;
mod shape;
