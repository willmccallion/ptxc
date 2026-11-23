//! Math programs against the f64 functions of the platform's libm: error in
//! ulps of the f32 result over a seeded sample always, over every f32 under
//! `--ignored` (run it with `--release`), and IEEE's special cases.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    reason = "test code: a panic is a test failure, and special cases compare exactly"
)]

mod accuracy;
mod power;
mod remainder;
mod specials;
