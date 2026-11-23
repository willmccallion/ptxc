//! PTX code generation from lowered loop IR kernels, through a typed PTX instruction tree.

mod ast;
mod print;
mod select;

pub use ast::PtxModule;
pub use select::{CodegenError, generate};
