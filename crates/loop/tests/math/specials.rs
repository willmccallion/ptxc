//! IEEE's special cases, and the boundaries of each program's range.

use ptxc_loop::math;

#[test]
fn exp_meets_ieee_special_cases() {
    let exp = |x| math::exp().eval(x);

    assert!(exp(f32::NAN).is_nan());
    assert_eq!(exp(f32::INFINITY), f32::INFINITY);
    assert_eq!(exp(f32::NEG_INFINITY), 0.0);
    assert_eq!(exp(0.0), 1.0);
    assert_eq!(exp(-0.0), 1.0);
    assert_eq!(exp(88.722_84), f32::INFINITY);
    assert!(exp(88.722_83).is_finite());
    assert_eq!(exp(-103.972_09), 0.0);
    assert_eq!(exp(-103.278_93), f32::from_bits(1));
}

#[test]
fn log_meets_ieee_special_cases() {
    let log = |x| math::log().eval(x);

    assert!(log(f32::NAN).is_nan());
    assert!(log(-1.0).is_nan());
    assert!(log(f32::NEG_INFINITY).is_nan());
    assert_eq!(log(f32::INFINITY), f32::INFINITY);
    assert_eq!(log(0.0), f32::NEG_INFINITY);
    assert_eq!(log(-0.0), f32::NEG_INFINITY);
    assert_eq!(log(1.0), 0.0);
    assert_eq!(log(f32::from_bits(1)), -103.278_93);
}
