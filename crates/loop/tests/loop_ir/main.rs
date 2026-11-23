//! Loop IR through its public API.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod compute_build;
mod compute_errors;
mod compute_types;
mod expressions;
mod schedule_properties;
mod schedule_rules;
mod schedule_support;
mod simplify_properties;
mod simplify_rules;
