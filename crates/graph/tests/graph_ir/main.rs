//! The graph IR through its public API: elementwise type rules, building,
//! rewriting, compaction, topological order and partitioning.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod builder;
mod common;
mod ops;
mod partition;
mod rewrite;
mod text;
mod traverse;
