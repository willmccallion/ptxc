//! GPU scheduling rules: tiling, thread mapping, shared-memory layout, tensor cores and software pipelining.

mod default;
mod pipeline;
mod record;
mod rule;
mod rules;
mod smem;
mod space;
mod split_k;
mod tiling;

pub use default::default_schedule;
pub use split_k::split_k;
