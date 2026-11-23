//! Lowered graphs, run stage by stage in the loop interpreter, against the
//! reference interpreter.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod activation;
mod cases;
mod elementwise;
mod harness;
mod norm;
mod reduce;
mod values;

/// Property-test cases: `PROPTEST_CASES` if set, else `default`.
pub(crate) fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES").ok().and_then(|text| text.parse().ok()).unwrap_or(default)
}
