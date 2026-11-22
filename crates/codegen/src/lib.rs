//! PTX code generation from lowered loop IR kernels, through a typed PTX instruction tree.

mod ast;
mod print;
mod select;
