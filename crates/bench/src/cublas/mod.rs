//! cuBLAS and cuBLASLt, loaded with dlopen, as the matrix-multiply
//! baseline.

pub(crate) mod gemm;
pub(crate) mod lt;
mod sys;
#[cfg(all(test, feature = "gpu"))]
mod testing;

pub(crate) use gemm::GemmError;
pub(crate) use sys::{Candidate, CublasError, Handle, LtHandle, Plan};
