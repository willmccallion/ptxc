//! Graph passes.

mod const_fold;
mod cse;
mod dce;
mod layout;

pub use cse::Cse;
pub use dce::Dce;
