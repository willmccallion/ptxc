//! Runtime: loads a Program onto a device and executes it eagerly or as a CUDA graph.

mod arena;
mod error;
mod evaluate;
mod executor;
mod io;
mod load;

pub use error::RuntimeError;
pub use executor::{Bound, Eager, Executor, Graphed};
pub use load::{LoadedProgram, load};
