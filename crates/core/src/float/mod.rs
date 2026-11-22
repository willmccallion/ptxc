//! Narrow float formats (f16, bf16, FP8 E4M3FN and E5M2) in software.
//!
//! All four share one `FloatFormat` description and one encoder: round to
//! nearest with ties to even, overflow checked after rounding, and FP8
//! overflow following ONNX `Cast`'s `saturate` attribute.

mod format;
mod types;

pub use format::Overflow;
pub use types::{Bf16, F8E4M3Fn, F8E5M2, F16};
