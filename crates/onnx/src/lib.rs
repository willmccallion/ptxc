//! ONNX frontend: parses a `ModelProto` into the graph IR, turning attributes into typed per-op structs.

mod attribute;
mod context;
mod error;
mod export;
mod fold;
mod import;
mod lowered;
mod model;
mod opset;
mod tensor;
mod value_info;

pub use error::{ImportError, ImportErrorKind, Unsupported};
pub use export::{ExportError, tensor_to_bytes};
pub use model::{ImportOptions, import_model, import_model_proto};
pub use tensor::tensor_from_bytes;
