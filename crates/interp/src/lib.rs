//! Reference interpreter for the graph IR: the golden model every compiled result is checked against.

mod env;
mod error;
mod eval;
mod extents;
mod interpreter;
mod movement;
mod numeric;

pub use error::{InterpError, Malformed};
pub use eval::evaluate as evaluate_op;
pub use interpreter::Interpreter;
