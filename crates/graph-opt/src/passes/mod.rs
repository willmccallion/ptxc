//! Graph passes.

mod const_fold;
mod cse;
mod dce;
mod layout;

pub use dce::Dce;
