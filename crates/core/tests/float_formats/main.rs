//! Narrow float formats: known encodings, decode/encode round trips and
//! rounding of f32 inputs against an independent oracle.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    reason = "test code: a panic is a test failure, and the oracle compares floats exactly on purpose"
)]

mod codec;
mod integers;
mod known_values;
mod oracle;
mod round_trip;
mod rounding;
