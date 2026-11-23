//! cuBLAS, loaded with dlopen, as the matrix-multiply baseline.

mod gemm;
mod lt;
mod sys;
#[cfg(all(test, feature = "gpu"))]
mod testing;
