//! Numeric conversions and their rounding rules.

use ptxc_core::convert::{f64_to_f32, int_to_f32, int_to_f64, truncate};

#[test]
fn narrowing_rounds_to_nearest_with_ties_to_even() {
    // Halfway between 1 and the next f32 (1 + 2^-23) ties down to even 1;
    // halfway between that and the one after ties up to even 1 + 2^-22.
    let first_tie = 1.0 + 2.0_f64.powi(-24);
    let second_tie = 1.0 + 2.0_f64.powi(-23) + 2.0_f64.powi(-24);

    assert_eq!(f64_to_f32(first_tie), 1.0);
    assert_eq!(f64_to_f32(second_tie), 1.0 + 2.0_f32.powi(-22));
    assert_eq!(f64_to_f32(1e300), f32::INFINITY);
    assert_eq!(int_to_f32((1 << 24) + 1), 16_777_216.0);
    assert_eq!(int_to_f64((1 << 53) + 1), 9_007_199_254_740_992.0);
    assert_eq!(int_to_f64(-3), -3.0);
}

#[test]
fn truncation_goes_toward_zero_and_saturates() {
    assert_eq!(truncate(-2.7), -2);
    assert_eq!(truncate(2.7), 2);
    assert_eq!(truncate(1e300), i64::MAX);
    assert_eq!(truncate(f64::NEG_INFINITY), i64::MIN);
    assert_eq!(truncate(f64::NAN), 0);
}
