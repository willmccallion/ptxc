//! Shared vocabulary: data types, float formats, shapes, sizes, typed indices, host tensors and the Evaluate trait.

mod dtype;
mod evaluate;
pub mod float;
mod index;
mod quant;
mod shape;
mod size;
mod tensor;

pub use dtype::DType;
pub use evaluate::Evaluate;
pub use index::{Idx, IndexVec};
pub use quant::{Affine, Granularity, QuantDType, QuantError, QuantParams, Scale, ZeroPoint};
pub use shape::{Axis, Dim, Rank, Shape, ShapeError};
pub use size::{Alignment, ByteOffset, ByteSize, ElementCount, SizeError};
pub use tensor::{HostTensor, TensorData, TensorError};
