//! Graph passes.

mod canonicalize;
mod const_fold;
mod cse;
mod dce;
mod layout;
mod simplify;

pub use canonicalize::Canonicalize;
pub use const_fold::ConstFold;
pub use cse::Cse;
pub use dce::Dce;
pub use simplify::Simplify;
