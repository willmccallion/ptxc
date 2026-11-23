//! Numeric conversions with a stated rounding rule, shared by the
//! interpreters so they round alike.

/// `value` rounded to the nearest `f32`, ties to even; out of range gives an
/// infinity.
#[expect(
    clippy::cast_possible_truncation,
    reason = "f64 to f32 rounds to nearest-even, as D19 asks"
)]
pub const fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

/// `value` truncated toward zero; out of range saturates and NaN gives 0,
/// as PTX's `cvt.rzi.sat` does.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Rust's float-to-int cast truncates and saturates"
)]
pub const fn truncate(value: f64) -> i64 {
    value as i64
}

/// An integer as the nearest `f64`, ties to even; exact below 2^53.
#[expect(clippy::cast_precision_loss, reason = "rounds to nearest-even, as ONNX Cast does")]
pub const fn int_to_f64(value: i64) -> f64 {
    value as f64
}

/// An integer as the nearest `f32`, rounded once.
#[expect(clippy::cast_precision_loss, reason = "rounds to nearest-even, as ONNX Cast does")]
pub const fn int_to_f32(value: i64) -> f32 {
    value as f32
}
