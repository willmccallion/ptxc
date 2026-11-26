//! Graph passes.

mod const_fold;
mod cse;
mod dce;
mod layout;

pub use const_fold::ConstFold;
pub use cse::Cse;
pub use dce::Dce;
