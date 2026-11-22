//! The graph IR through its public API: elementwise type rules, building,
//! rewriting, compaction and topological order.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod builder;
mod common;
mod ops;
mod rewrite;
mod text;
mod traverse;
