//! The element type arithmetic runs in: floats narrower than f32 widen to
//! f32 and round back once, as `ptxc-interp` computes (D19) and as GPU
//! kernels do outside tensor-core paths.

use ptxc_core::DType;
use ptxc_core::float::Overflow;
use ptxc_loop::TensorExpr;

/// The type arithmetic on `dtype` values runs in.
pub(crate) const fn arithmetic_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::Bf16 | DType::F8E4M3Fn | DType::F8E5M2 => DType::F32,
        DType::F32 | DType::I8 | DType::U8 | DType::I32 | DType::I64 | DType::Bool => dtype,
    }
}

/// `expr`, of type `dtype`, in the type its arithmetic runs in.
pub(crate) fn promote(expr: TensorExpr, dtype: DType) -> TensorExpr {
    convert(expr, dtype, arithmetic_dtype(dtype))
}

/// `expr`, computed in `dtype`'s arithmetic type, rounded back to `dtype`.
pub(crate) fn demote(expr: TensorExpr, dtype: DType) -> TensorExpr {
    convert(expr, arithmetic_dtype(dtype), dtype)
}

/// `expr` converted from `from` to `to`; a float overflowing `to` becomes
/// an infinity, as IEEE 754 rounding gives.
pub(crate) fn convert(expr: TensorExpr, from: DType, to: DType) -> TensorExpr {
    if from == to {
        return expr;
    }
    TensorExpr::Cast { to, overflow: Overflow::NonFinite, value: Box::new(expr) }
}
