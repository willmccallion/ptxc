//! A GEMM problem as the sweep defines it: `C = A B` in row-major storage,
//! A of m by k and B of k by n, each stored as is (N) or transposed (T),
//! with f32 accumulation (BENCHMARKING.md, GEMM against cuBLAS).

use ptxc_core::{ByteSize, DType};

/// The multiplicands' element type. Floats accumulate in f32, and C has
/// their type, except E4M3, whose C is f32; int8 accumulates and writes
/// i32.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GemmType {
    F32,
    F16,
    Bf16,
    E4m3,
    I8,
}

/// How A and B are stored, in that order: N as is, T transposed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Layout {
    Nn,
    Nt,
    Tn,
}

/// The extents of `C = A B`: C is m by n, the reduction k long.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct GemmShape {
    pub(crate) m: u32,
    pub(crate) n: u32,
    pub(crate) k: u32,
}

/// One GEMM: its shape, element type and layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct GemmProblem {
    pub(crate) shape: GemmShape,
    pub(crate) dtype: GemmType,
    pub(crate) layout: Layout,
}

impl GemmType {
    /// A and B's element type.
    pub(crate) const fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::Bf16,
            Self::E4m3 => DType::F8E4M3Fn,
            Self::I8 => DType::I8,
        }
    }

    /// C's element type.
    pub(crate) const fn output(self) -> DType {
        match self {
            Self::F32 | Self::E4m3 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::Bf16,
            Self::I8 => DType::I32,
        }
    }
}

impl Layout {
    /// Whether A is stored transposed.
    pub(crate) const fn a_transposed(self) -> bool {
        matches!(self, Self::Tn)
    }

    /// Whether B is stored transposed.
    pub(crate) const fn b_transposed(self) -> bool {
        matches!(self, Self::Nt)
    }
}

impl GemmProblem {
    /// The bytes A, B and C take, or `None` past `u64`.
    pub(crate) fn sizes(self) -> Option<[ByteSize; 3]> {
        let GemmShape { m, n, k } = self.shape;
        let bytes = |rows: u32, columns: u32, dtype: DType| {
            let elements = u64::from(rows).checked_mul(u64::from(columns))?;
            elements.checked_mul(dtype.element_size().get()).map(ByteSize::new)
        };
        let (input, output) = (self.dtype.dtype(), self.dtype.output());
        Some([bytes(m, k, input)?, bytes(k, n, input)?, bytes(m, n, output)?])
    }
}
