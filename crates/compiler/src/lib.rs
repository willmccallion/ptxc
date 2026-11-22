//! Compiler driver: runs the pipeline from ONNX bytes to a Program, including static memory planning.

mod compilation;
mod emit;
mod memory_plan;
mod options;
mod tuning_db;
