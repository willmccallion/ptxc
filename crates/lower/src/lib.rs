//! Lowering from graph IR fused groups to loop IR compute definitions.

mod builder;
mod compose;
mod epilogue;
mod error;
mod group;
mod ops;
mod precision;
mod stage;

pub use error::LowerError;
pub use group::lower;
pub use stage::{Binding, IntermediateId, LoweredGroup, Stage, TableId};
