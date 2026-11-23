//! Shared vocabulary: data types, float formats, shapes, sizes, typed indices, host tensors, kernel names, time and the Evaluate trait.

pub mod arith;
pub mod convert;
mod dtype;
mod evaluate;
pub mod float;
mod index;
mod kernel_name;
mod quant;
mod shape;
mod size;
mod tensor;
mod tensor_bytes;
mod time;

pub use dtype::DType;
pub use evaluate::Evaluate;
pub use index::{Idx, IndexVec};
pub use kernel_name::KernelName;
pub use quant::{Affine, Granularity, QuantDType, QuantError, QuantParams, Scale, ZeroPoint};
pub use shape::{Axis, Dim, Rank, Shape, ShapeError};
pub use size::{Alignment, ByteOffset, ByteSize, ElementCount, SizeError};
pub use tensor::{HostTensor, TensorData, TensorError};
pub use tensor_bytes::BytesError;
pub use time::Microseconds;
