//! Graph passes.

mod canonicalize;
mod const_fold;
mod cse;
mod dce;
mod layout;

pub use canonicalize::Canonicalize;
pub use const_fold::ConstFold;
pub use cse::Cse;
pub use dce::Dce;
