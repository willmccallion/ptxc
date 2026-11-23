//! A text format for graphs, used by golden tests and for reading IR by hand:
//!
//! ```text
//! graph(%x: f32[2, 3]) {
//!   %c = const f32[3] [0.5, 1.0, -2.0]
//!   %y = add %x, %c : f32[2, 3]
//!   return %y
//! }
//! ```
//!
//! Constants with more than 16 elements print as `<elided 0x...>`, a content
//! hash, and parse from tensors the caller supplies.

mod attr;
mod cursor;
mod element_text;
mod error;
mod lexer;
mod ops;
mod parse;
mod print;
mod tensor_text;

pub use error::{ParseError, ParseErrorKind, Position};
pub use parse::parse;
pub use print::print;

use crate::tensor_type::TensorType;

/// A tensor type as the text format writes it, such as `f32[2, 3]`.
pub fn format_type(ty: &TensorType) -> String {
    let mut out = String::new();
    tensor_text::write_type(&mut out, ty);
    out
}
