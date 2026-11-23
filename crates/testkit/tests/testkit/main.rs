//! The testkit through its public API.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod compare;
mod data;
mod golden;
mod onnx_tests;
