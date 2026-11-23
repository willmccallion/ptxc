//! The importer through its public API, on small models built in code:
//! tensors, attributes, model-level rules, and each family's canonical form.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod builders;
mod model;
mod ops;
mod tensors;
