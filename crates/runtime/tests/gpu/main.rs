//! Kernels ptxc emits, run on a real GPU and compared with the loop
//! interpreter, which they must match bit for bit, NaN payloads aside, or
//! within tolerance where they multiply on tensor cores (D60).
//! Compiled only with `--features gpu`; without a GPU these fail rather
//! than skip.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod convolutions;
mod device;
mod families;
mod fragments;
mod fused;
mod graphed;
mod kernels;
mod products;
mod programs;
mod split_products;
mod tensor_products;
