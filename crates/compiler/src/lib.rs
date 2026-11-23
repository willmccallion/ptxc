//! Compiler driver: runs the pipeline from ONNX bytes to a Program, including static memory planning.

mod compilation;
mod emit;
mod error;
mod memory_plan;
mod options;
mod tuning_db;

pub use compilation::{Compilation, Imported, Lowered, Optimized, Partitioned, Scheduled};
pub use error::CompileError;
