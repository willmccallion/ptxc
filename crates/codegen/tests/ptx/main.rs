//! Emitted PTX: golden text for representative kernels, and every module
//! assembled by ptxas (D11).

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod fragments;
mod fused;
mod kernels;
mod ptxas;
mod unrolled;
mod vector;

use ptxc_codegen::{PtxModule, generate};
use ptxc_loop::ComputeDef;
use ptxc_target::{SmVersion, Target, preset};
use std::sync::Arc;

pub(crate) fn target() -> Target {
    preset(SmVersion::SM_89).unwrap()
}

/// The module for `compute` under its default schedule.
pub(crate) fn emitted(compute: Arc<ComputeDef>) -> PtxModule {
    let target = target();
    let kernel = ptxc_schedule::default_schedule(compute, &target).unwrap().lower();
    generate(&kernel, &target).unwrap()
}
