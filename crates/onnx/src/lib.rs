//! ONNX frontend: parses a `ModelProto` into the graph IR, turning attributes into typed per-op structs.

mod attribute;
mod error;
mod import;
mod model;
mod opset;
mod tensor;
