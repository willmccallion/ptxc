//! Shape ops: each output element is an input element at a computed
//! position, so every element type is handled alike.

mod pad;
mod slice;
mod view;

pub(in crate::eval) use pad::{pad, trilu};
pub(in crate::eval) use slice::{slice, split};
pub(in crate::eval) use view::{concat, expand, tile, transpose};
