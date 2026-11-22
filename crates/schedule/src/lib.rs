//! GPU scheduling rules: tiling, thread mapping, shared-memory layout, tensor cores and software pipelining.

mod default;
mod pipeline;
mod record;
mod rule;
mod rules;
mod smem;
mod space;
mod tiling;
