//! Lowered graphs, run stage by stage in the loop interpreter, against the
//! reference interpreter.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod activation;
mod attention;
mod cases;
mod elementwise;
mod gather;
mod harness;
mod matmul;
mod norm;
mod quant;
mod reduce;
mod resize;
mod shape;
mod values;
mod window;

/// Property-test cases: `PROPTEST_CASES` if set, else `default`.
pub(crate) fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES").ok().and_then(|text| text.parse().ok()).unwrap_or(default)
}
