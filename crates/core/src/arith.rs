//! Integer arithmetic both interpreters share, so they agree exactly.

use crate::convert::{int_to_f64, truncate};

/// `base ^ exponent` by repeated squaring, wrapping. A negative exponent
/// gives the float power truncated toward zero.
pub fn wrapping_pow(base: i64, exponent: i64) -> i64 {
    // NOTE: NumPy rejects negative integer exponents and ONNX's spec is
    // silent; ONNX Runtime computes them in floating point and truncates.
    let Ok(mut remaining) = u64::try_from(exponent) else {
        return truncate(int_to_f64(base).powf(int_to_f64(exponent)));
    };
    let (mut result, mut square) = (1_i64, base);
    while remaining > 0 {
        if remaining & 1 == 1 {
            result = result.wrapping_mul(square);
        }
        square = square.wrapping_mul(square);
        remaining >>= 1;
    }
    result
}
