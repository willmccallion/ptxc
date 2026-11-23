//! The C entry points of cuBLAS and cuBLASLt, and the calls into them: the
//! only unsafe code in `ptxc-bench` (D55). Each function's type transcribes
//! its declaration in the cuBLAS 13 headers.

mod cublas;
mod cublaslt;

pub(super) use cublas::{ComputeType, DataType, GemmCall, Operand, Operation};
pub(crate) use cublas::{CublasError, Handle};
pub(crate) use cublaslt::{Candidate, LtHandle, Plan};
pub(super) use cublaslt::{LtProblem, MatrixLayout};
