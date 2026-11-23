//! Elementwise ops: unary, binary and variadic ops, `Where`, and casts.

mod binary;
mod cast;
mod unary;
mod variadic;

pub(super) use binary::binary;
pub(super) use cast::cast;
pub(super) use unary::unary;
pub(super) use variadic::{select, variadic};
